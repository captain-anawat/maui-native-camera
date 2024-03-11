using FaceDetectionLib.PrePostProcessing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceDetectionLib.OnnxModels.EyeBlink;

internal class EyeBlinkSample : VisionSampleBase<EyeBlinkImageProcessor>
{
    public string[] EyeValue;
    public const string Identifier = "EyeBlink";
    public const string ModelFilename = "eye_blink_cnn.onnx";
    public EyeBlinkSample()
        : base(Identifier, ModelFilename) { }
    public (SKRect, SKRect) GetEyeRectangles(SKPoint[] points)
    {
        var leftEye = points.GetLeftEye().GetRectangle();
        var rightEye = points.GetRightEye().GetRectangle();
        return (leftEye, rightEye);
    }

    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        // do initial resize maintaining the aspect ratio so the smallest size is 800. this is arbitrary and 
        // chosen to be a good size to dispay to the user with the results
        using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f))
                                          .ConfigureAwait(false);

        // do the preprocessing to resize the image to the 34x26 with the model expects. 
        // NOTE: this does not maintain the aspect ratio but works well enough with this particular model.
        //       it may be better in other scenarios to resize and crop to convert the original image to a
        //       34x26 image.
        using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image))
                                                .ConfigureAwait(false);

        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage))
                               .ConfigureAwait(false);

        // Run the model
        var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height))
                                    .ConfigureAwait(false);

        throw new NotImplementedException();
    }

    public async Task<EyeBlinkPrediction> OnProcessPredictionAsync(byte[] image)
    {
        // do initial resize maintaining the aspect ratio so the smallest size is 800. this is arbitrary and 
        // chosen to be a good size to dispay to the user with the results
        using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f))
                                          .ConfigureAwait(false);

        // do the preprocessing to resize the image to the 34x26 with the model expects. 
        // NOTE: this does not maintain the aspect ratio but works well enough with this particular model.
        //       it may be better in other scenarios to resize and crop to convert the original image to a
        //       34x26 image.
        using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image))
                                                .ConfigureAwait(false);

        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage))
                               .ConfigureAwait(false);

        // Run the model
        var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height))
                                    .ConfigureAwait(false);

        return predictions.FirstOrDefault();
    }

    List<EyeBlinkPrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
    {
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata = Session.InputMetadata;
        string name = inputMetadata.Keys.ToArray()[0];

        // Setup inputs. Names used must match the input names in the model. 
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, input) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);
        DisposableNamedOnnxValue[] data = results.ToArray();
        var num = data.Length;
        var eyeTensor = data[num - 1].AsTensor<float>().ToArray();
        EyeValue = ToString(eyeTensor);
        return new List<EyeBlinkPrediction> {
            new EyeBlinkPrediction{
                EyeValue = eyeTensor
            }
        };

        string[] ToString(float[] tensor)
        {
            var value = Math.Round(tensor[0], 1);
            return new string[] { value.ToString() };
        }
    }
}

