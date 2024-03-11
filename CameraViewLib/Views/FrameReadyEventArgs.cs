namespace CameraViewLib.Views;

public class FrameReadyEventArgs : EventArgs
{
    public enum PixelFormat
    {
        GRAYSCALE,
        RGB888,
        BGR888,
        RGBA8888,
        BGRA8888,
    }
    public FrameReadyEventArgs(byte[] buffer, int width, int height, int stride, PixelFormat pixelFormat)
    {
        Buffer = buffer;
        Width = width;
        Height = height;
        Stride = stride;
        Format = pixelFormat;
    }

    public byte[] Buffer { get; private set; }
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Stride { get; private set; }
    public PixelFormat Format { get; private set; }
}

