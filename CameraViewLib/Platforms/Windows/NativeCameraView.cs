﻿using CameraViewLib.Views;
using Microsoft.UI.Xaml.Controls;
using System.Collections.Concurrent;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Devices.Enumeration;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.Core;
using static CameraViewLib.Views.CameraInfo;
using Panel = Windows.Devices.Enumeration.Panel;

namespace CameraViewLib.Platforms.Windows;

public class NativeCameraView : UserControl, IDisposable
{
    private Microsoft.UI.Xaml.FlowDirection flowDirection = Microsoft.UI.Xaml.FlowDirection.LeftToRight;
    private Thread thread;
    private static object lockObject = new object();
    private ConcurrentQueue<SoftwareBitmap> _bitmapQueue = new ConcurrentQueue<SoftwareBitmap>();
    private volatile bool isCapturing;
    private readonly MediaPlayerElement mediaElement;
    private MediaCapture mediaCapture;
    private MediaFrameSource frameSource;
    private MediaFrameReader frameReader;
    private List<MediaFrameSourceGroup> sourceGroups;
    private bool started = false, initiated = false;
    private readonly CameraView cameraView;

    public NativeCameraView(CameraView cameraView)
    {
        this.cameraView = cameraView;
        mediaElement = new MediaPlayerElement
        {
            HorizontalAlignment = Microsoft.UI.Xaml.HorizontalAlignment.Stretch,
            VerticalAlignment = Microsoft.UI.Xaml.VerticalAlignment.Stretch
        };
        Content = mediaElement;
        InitCameras();
    }

    private void ProcessFrames()
    {
        while (isCapturing)
        {
            SoftwareBitmap bitmap;
            bool ret = _bitmapQueue.TryDequeue(out bitmap);
            if (ret)
            {
                //int pixelSize = bitmap.PixelWidth * bitmap.PixelHeight * 4; // Assuming 32bpp (Rgba8)
                //byte[] buffer = new byte[pixelSize];
                byte[] buffer = new byte[bitmap.PixelWidth * bitmap.PixelHeight];
                bitmap.CopyToBuffer(buffer.AsBuffer());
                cameraView.NotifyFrameReady(buffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.PixelWidth, FrameReadyEventArgs.PixelFormat.GRAYSCALE);
                bitmap.Dispose();
            }
        }
    }

    private void Create()
    {
        lock (lockObject)
        {
            isCapturing = true;
            thread = new Thread(new ThreadStart(ProcessFrames));
            thread.Start();
        }
    }

    private void InitCameras()
    {
        if (!initiated)
        {
            try
            {
                var devices = DeviceInformation.FindAllAsync(DeviceClass.VideoCapture).GetAwaiter().GetResult();
                var allSourceGroups = MediaFrameSourceGroup.FindAllAsync().GetAwaiter().GetResult();
                sourceGroups = allSourceGroups.Where(g => g.SourceInfos.Any(s => s.SourceKind == MediaFrameSourceKind.Color &&
                                                                                    (s.MediaStreamType == MediaStreamType.VideoPreview || s.MediaStreamType == MediaStreamType.VideoRecord))
                                                                                    && g.SourceInfos.All(sourceInfo => devices.Any(device => device.Id == sourceInfo.DeviceInformation.Id))).ToList();
                cameraView.Cameras.Clear();
                foreach (var sourceGroup in sourceGroups)
                {
                    Position position = Position.Unknown;
                    var device = devices.FirstOrDefault(device => device.Id == sourceGroup.Id);
                    if (device != null)
                    {
                        if (device.EnclosureLocation != null)
                            position = device.EnclosureLocation.Panel switch
                            {
                                Panel.Front => Position.Front,
                                Panel.Back => Position.Back,
                                _ => Position.Unknown
                            };
                    }

                    var camInfo = new CameraInfo
                    {
                        Name = sourceGroup.DisplayName,
                        DeviceId = sourceGroup.Id,
                        Pos = position,
                        AvailableResolutions = new()
                    };
                    foreach (var profile in MediaCapture.FindAllVideoProfiles(sourceGroup.Id))
                    {
                        foreach (var recordMediaP in profile.SupportedRecordMediaDescription)
                        {
                            if (!camInfo.AvailableResolutions.Any(s => s.Width == recordMediaP.Width && s.Height == recordMediaP.Height))
                                camInfo.AvailableResolutions.Add(new(recordMediaP.Width, recordMediaP.Height));
                        }
                    }
                    cameraView.Cameras.Add(camInfo);
                }

                initiated = true;
                cameraView.UpdateCameras();

                Create();
            }
            catch
            {
            }
        }
    }


    internal async Task<Status> StartCameraAsync()
    {
        Status result = Status.Unavailable;

        if (initiated)
        {
            if (started) await StopCameraAsync();
            if (cameraView.Camera != null)
            {
                started = true;
                mediaCapture = new MediaCapture();
                try
                {
                    await mediaCapture.InitializeAsync(new MediaCaptureInitializationSettings
                    {
                        SourceGroup = sourceGroups.First(source => source.Id == cameraView.Camera.DeviceId),
                        MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                        StreamingCaptureMode = StreamingCaptureMode.Video
                    });
                    frameSource = mediaCapture.FrameSources.FirstOrDefault(source => source.Value.Info.MediaStreamType == MediaStreamType.VideoRecord
                                                                                          && source.Value.Info.SourceKind == MediaFrameSourceKind.Color).Value;
                    if (frameSource != null)
                    {
                        MediaFrameFormat frameFormat;
                        frameFormat = frameSource.SupportedFormats.OrderByDescending(f => f.VideoFormat.Width * f.VideoFormat.Height).FirstOrDefault();

                        if (frameFormat != null)
                        {
                            await frameSource.SetFormatAsync(frameFormat);
                            mediaElement.AutoPlay = true;
                            mediaElement.Source = MediaSource.CreateFromMediaFrameSource(frameSource);
                            mediaElement.FlowDirection = flowDirection;

                            frameReader = await mediaCapture.CreateFrameReaderAsync(frameSource);
                            frameReader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;
                            if (frameReader != null)
                            {
                                frameReader.FrameArrived += OnFrameAvailable;
                                var status = await frameReader.StartAsync();
                                if (status == MediaFrameReaderStartStatus.Success)
                                {
                                    result = Status.Available;
                                }
                            }

                        }
                    }
                }
                catch
                {
                }
            }

            if (result != Status.Available && mediaCapture != null)
            {
                if (frameReader != null)
                {
                    frameReader.FrameArrived -= OnFrameAvailable;
                    frameReader.Dispose();
                    frameReader = null;
                }
                mediaCapture.Dispose();
                mediaCapture = null;
            }
        }

        return result;
    }

    private void OnFrameAvailable(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        var frame = sender.TryAcquireLatestFrame();
        if (frame == null) return;

        SoftwareBitmap bitmap = frame.VideoMediaFrame.SoftwareBitmap;
        if (_bitmapQueue.Count == 2) ClearQueue();
        SoftwareBitmap grayscale = SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Gray8, BitmapAlphaMode.Ignore);
        _bitmapQueue.Enqueue(grayscale);
        bitmap.Dispose();
    }

    internal async Task<Status> StopCameraAsync()
    {
        Status result = Status.Available;
        if (initiated)
        {
            try
            {
                if (frameReader != null)
                {
                    await frameReader.StopAsync();
                    frameReader.FrameArrived -= OnFrameAvailable;
                    frameReader?.Dispose();
                    frameReader = null;
                }
                mediaElement.Source = null;
                if (mediaCapture != null)
                {
                    mediaCapture.Dispose();
                    mediaCapture = null;
                }
            }
            catch
            {
                result = Status.Unavailable;
            }
        }
        else
            result = Status.Unavailable;
        started = false;
        Destroy();
        return result;
    }
    internal void DisposeControl()
    {
        if (started) StopCameraAsync().Wait();
        Dispose();
    }

    public void Dispose()
    {
        Destroy();
        StopCameraAsync().Wait();
    }

    public void Destroy()
    {
        lock (lockObject)
        {
            if (thread != null)
            {
                isCapturing = false;

                thread.Join();
                thread = null;
            }

            ClearQueue();
        }
    }

    private void ClearQueue()
    {
        while (_bitmapQueue.Count > 0)
        {
            SoftwareBitmap bitmap;
            _bitmapQueue.TryDequeue(out bitmap);
            bitmap.Dispose();
        }
    }
}
