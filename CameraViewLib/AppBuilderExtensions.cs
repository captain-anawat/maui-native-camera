namespace CameraViewLib;

public static class AppBuilderExtensions
{
    public static MauiAppBuilder UseNativeCameraView(this MauiAppBuilder builder)
    {
        builder.ConfigureMauiHandlers(h =>
        {
            h.AddHandler(typeof(CameraView), typeof(CameraViewHandler));
        });
        return builder;
    }
}
