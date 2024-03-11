using FaceDetectionLib.PrePostProcessing;
using SkiaSharp;

namespace FaceDetectionLib;

internal static class SkiaSharpSupportModel
{
    public static SKPoint[] GetLeftEye(this SKPoint[] points)
    {
        if (points.Length != 68)
        {
            throw new ArgumentException("The number of face points must be 68.");
        }

        SKPoint[] array = new SKPoint[6];
        for (int i = 0; i < 6; i++)
        {
            array[i] = points[i + 36];
        }

        return array;
    }
    public static SKPoint[] GetRightEye(this SKPoint[] points)
    {
        if (points.Length != 68)
        {
            throw new ArgumentException("The number of face points must be 68.");
        }

        SKPoint[] array = new SKPoint[6];
        for (int i = 0; i < 6; i++)
        {
            array[i] = points[i + 42];
        }

        return array;
    }

    public static SKRect GetRectangle(this SKPoint[] points)
    {
        var num = points.Length;
        var num2 = float.MaxValue;
        var num3 = float.MaxValue;
        var num4 = float.MinValue;
        var num5 = float.MinValue;
        for (int i = 0; i < num; i++)
        {
            var x = points[i].X;
            var y = points[i].Y;
            if (x < num2)
            {
                num2 = x;
            }

            if (y < num3)
            {
                num3 = y;
            }

            if (x > num4)
            {
                num4 = x;
            }

            if (y > num5)
            {
                num5 = y;
            }
        }
        var kx = 0f;
        var ky = 0.5f;
        var width = num4 - num2;
        var height = num5 - num3;
        var scaleX = width * kx;
        var scaleY = height * ky;
        return SKRect.Create(num2 - scaleX / 2, num3 - scaleY / 2, width + scaleX, height + scaleY);
    }

    public static byte[] SKBitmapCrop(this SKBitmap bitmap, PredictionBox box)
    {
        using var pixmap = new SKPixmap(bitmap.Info, bitmap.GetPixels());
        SKRectI rectI = new SKRectI((int)box.Xmin, (int)box.Ymin,
            (int)box.Xmax, (int)box.Ymax);

        var subset = pixmap.ExtractSubset(rectI);

        using var data = subset.Encode(SKPngEncoderOptions.Default);
        return data.ToArray();
    }
}
