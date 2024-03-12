using CameraViewLib.Views;
using FaceDetectionLib;
using FaceDetectionLib.OnnxModels.UltraFace;
using SkiaSharp;
using System.Runtime.InteropServices;
using static CameraViewLib.Views.FrameReadyEventArgs;

namespace CameraViewLib;

public partial class CameraViewPage : ContentPage
{
    IVisionSample _ultraface;
    IVisionSample Ultraface => _ultraface ??= new UltrafaceSample();
    bool saveImage = false;
    bool cameraReady = true;
    public CameraViewPage()
    {
        InitializeComponent();
    }
    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        cameraView.ShowCameraView = false;
    }
    private async void cameraView_FrameReady(object sender, FrameReadyEventArgs e)
    {
        if (!cameraReady) return;
        byte[] buffer = e.Buffer;
        int width = e.Width;
        int height = e.Height;
        int stride = e.Stride;
        PixelFormat format = e.Format;

        SKImageInfo info = new SKImageInfo(width, height, SKColorType.Gray8, SKAlphaType.Premul);

        // Create the SKBitmap
        SKBitmap bitmap = new SKBitmap(info);
        bitmap.SetPixels((Marshal.UnsafeAddrOfPinnedArrayElement(buffer, 0)));

        // Save the data to a file
        using SKImage image = SKImage.FromBitmap(bitmap);
        SKData encoded = image.Encode();
        var bytes = encoded.ToArray();
        try
        {
            var result = await Ultraface.ProcessImageAsync(bytes);
            MainThread.BeginInvokeOnMainThread(() =>
            {
                OutputImage.Source = ImageSource.FromStream(() => new MemoryStream(result.Image));
                caption.Text = result.Caption;
            });

            //// process image
            //if (index < 60 && saveImage)
            //{
            //    index++;

            //    using SKData data = image.Encode(SKEncodedImageFormat.Jpeg, 100);
            //    var appDataDirectory = FileSystem.AppDataDirectory;

            //    // Combine the app's data directory path with your file name
            //    var filePath = Path.Combine(appDataDirectory, $"yourfile{index}.jpg");

            //    using FileStream streamA = File.OpenWrite(filePath);
            //    data.SaveTo(streamA);
            //}
        }
        catch
        {
            MainThread.BeginInvokeOnMainThread(() =>
            {
                OutputImage.Source = ImageSource.FromStream(() => new MemoryStream(bytes));
            });
        }
    }

}