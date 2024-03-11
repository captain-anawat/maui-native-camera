using System.Collections.ObjectModel;
using CameraViewLib.Views;
using static CameraViewLib.Views.CameraInfo;

namespace CameraViewLib;

public class CameraView : View
{
    public static readonly BindableProperty CamerasProperty = BindableProperty.Create(nameof(Cameras), typeof(ObservableCollection<CameraInfo>), typeof(CameraView), new ObservableCollection<CameraInfo>());
    public static readonly BindableProperty CameraProperty = BindableProperty.Create(nameof(Camera), typeof(CameraInfo), typeof(CameraView), null);
    public static readonly BindableProperty ShowCameraViewProperty = BindableProperty.Create(nameof(ShowCameraView), typeof(bool), typeof(CameraView), false, propertyChanged: ShowCameraViewChanged);
    public event EventHandler<FrameReadyEventArgs> FrameReady;
    public string BarcodeParameters { get; set; } = string.Empty;

    public ObservableCollection<CameraInfo> Cameras
    {
        get { return (ObservableCollection<CameraInfo>)GetValue(CamerasProperty); }
        set { SetValue(CamerasProperty, value); }
    }

    public CameraInfo Camera
    {
        get { return (CameraInfo)GetValue(CameraProperty); }
        set { SetValue(CameraProperty, value); }
    }

    public bool ShowCameraView
    {
        get { return (bool)GetValue(ShowCameraViewProperty); }
        set { SetValue(ShowCameraViewProperty, value); }
    }

    public void NotifyFrameReady(byte[] buffer, int width, int height, int stride, FrameReadyEventArgs.PixelFormat format)
    {
        if (FrameReady != null)
        {
            FrameReady(this, new FrameReadyEventArgs(buffer, width, height, stride, format));
        }
    }

    private static async void ShowCameraViewChanged(BindableObject bindable, object oldValue, object newValue)
    {
        if (oldValue != newValue && bindable is CameraView control)
        {
            try
            {
                if ((bool)newValue)
                    await control.StartCameraAsync();
                else
                    await control.StopCameraAsync();
            }
            catch { }
        }
    }

    public async Task<Status> StartCameraAsync()
    {
        Status result = Status.Unavailable;
        if (Camera != null)
        {
            if (Handler != null && Handler is CameraViewHandler handler)
            {
                result = await handler.StartCameraAsync();
            }
        }

        return result;
    }

    public async Task<Status> StopCameraAsync()
    {
        Status result = Status.Unavailable;
        if (Handler != null && Handler is CameraViewHandler handler)
        {
            result = await handler.StopCameraAsync();
        }
        return result;
    }

    public void UpdateCameras()
    {
        Task.Run(() => {

            MainThread.BeginInvokeOnMainThread(() => {
                if (Cameras.Count > 0)
                {
                    Camera = Cameras.First();
                    ShowCameraView = true;
                    OnPropertyChanged(nameof(ShowCameraView));
                }
            });

        });
    }
    public static async Task<bool> RequestPermissions()
    {
        var status = await Permissions.CheckStatusAsync<Permissions.Camera>();
        if (status != PermissionStatus.Granted)
        {
            status = await Permissions.RequestAsync<Permissions.Camera>();
            if (status != PermissionStatus.Granted) return false;
        }
        return true;
    }
}

