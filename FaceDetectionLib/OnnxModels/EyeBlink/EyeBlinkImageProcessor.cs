using FaceDetectionLib.PrePostProcessing;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceDetectionLib.OnnxModels.EyeBlink;

internal class EyeBlinkImageProcessor : SkiaSharpImageProcessor<EyeBlinkPrediction, float>
{
    const int RequiredWidth = 34;
    const int RequiredHeight = 26;
    protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
        => sourceImage.Resize(new SKImageInfo(RequiredWidth, RequiredHeight), SKFilterQuality.Medium);

    protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
    {
        SKImageInfo info = image.Info;
        if (info.ColorType != SKColorType.Bgra8888 || info.AlphaType != SKAlphaType.Premul)
        {
            throw new ArgumentException("Image must be in BGRA8888 format");
        }

        SKBitmap resizedImage = new SKBitmap(34, 26);
        using (SKCanvas canvas = new SKCanvas(resizedImage))
        {
            canvas.Clear(SKColors.Transparent);
            canvas.DrawBitmap(image, new SKRect(0, 0, 34, 26));
        }

        var array = new float[3][,];
        for (int i = 0; i < 3; i++)
        {
            array[i] = GetChannelAsArray(resizedImage, i);
        }

        var array2 = new int[4] { 1, 26, 34, 1 };
        var array3 = ComputeAverage(array);
        return new DenseTensor<float>(array3, array2);

        float[,] GetChannelAsArray(SKBitmap image, int channelIndex)
        {
            int width = image.Width;
            int height = image.Height;
            float[,] channelArray = new float[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    SKColor pixel = image.GetPixel(x, y);
                    channelArray[y, x] = GetChannelValue(pixel, channelIndex);
                }
            }
            return channelArray;
        }

        float GetChannelValue(SKColor color, int channelIndex)
        {
            switch (channelIndex)
            {
                case 0: return color.Red / 255f;
                case 1: return color.Green / 255f;
                case 2: return color.Blue / 255f;
                default: throw new ArgumentOutOfRangeException(nameof(channelIndex));
            }
        }

        float[] ComputeAverage(float[][,] channels)
        {
            int height = channels[0].GetLength(0);
            int width = channels[0].GetLength(1);
            float[] averageArray = new float[height * width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0f;
                    for (int i = 0; i < 3; i++)
                    {
                        sum += channels[i][y, x];
                    }
                    averageArray[y * width + x] = sum / 3f;
                }
            }

            return averageArray;
        }
    }

}

