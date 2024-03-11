using FaceDetectionLib.OnnxModels.EyeBlink;
using FaceDetectionLib.PrePostProcessing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceDetectionLib.OnnxModels.FaceLandmark;

internal class FaceLandmarkSample : VisionSampleBase<FaceLandmarkImageProcessor>
{
    public const string Identifier = "FaceLandmark";
    public const string ModelFilename = "landmarks_68_pfld.onnx";
    float eyeL;
    float eyeR;
    FaceLandmarkPrediction savesolt;
    SKBitmap saveSourceImage;
    EyeBlinkSample _eyeblink;
    EyeBlinkSample Eyeblink => _eyeblink;
    public FaceLandmarkSample()
        : base(Identifier, ModelFilename)
    {
        _eyeblink ??= new EyeBlinkSample();
    }

    public (SKBitmap, FaceLandmarkPrediction) GetSavePrediction()
    {
        return (saveSourceImage, savesolt);
    }
    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        // do initial resize maintaining the aspect ratio so the smallest size is 800. this is arbitrary and 
        // chosen to be a good size to dispay to the user with the results
        using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f))
                                          .ConfigureAwait(false);

        // do the preprocessing to resize the image to the 112x112 with the model expects. 
        // NOTE: this does not maintain the aspect ratio but works well enough with this particular model.
        //       it may be better in other scenarios to resize and crop to convert the original image to a
        //       112x112 image.
        using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image))
                                                .ConfigureAwait(false);

        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage))
                               .ConfigureAwait(false);

        // Run the model
        var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height))
                                    .ConfigureAwait(false);

        if (predictions is not null)
        {
            // Pointing Eyes position
            var (lefteye, righteye) = Eyeblink.GetEyeRectangles(predictions[0].Points);
            var leftEyeBox = new PredictionBox(lefteye.Left, lefteye.Top, lefteye.Right, lefteye.Bottom);
            var rightEyeBox = new PredictionBox(righteye.Left, righteye.Top, righteye.Right, righteye.Bottom);
            var leftEyeImage = sourceImage.SKBitmapCrop(leftEyeBox);
            var rightEyeImage = sourceImage.SKBitmapCrop(rightEyeBox);
            var leftblinkValue = await Eyeblink.OnProcessPredictionAsync(leftEyeImage);
            eyeL = leftblinkValue.EyeValue[0];
            var rightblinkvValue = await Eyeblink.OnProcessPredictionAsync(rightEyeImage);
            eyeR = rightblinkvValue.EyeValue[0];
            //return new ImageProcessingResult(rightEyeImage);
        }
        savesolt = predictions[0];
        saveSourceImage = sourceImage;
        var pre = new List<FaceLandmarkPrediction> {
        new FaceLandmarkPrediction{
            Points =predictions[0].Points.GetLeftEye().Union(predictions[0].Points.GetRightEye()).ToArray()
        }
    };
        // Draw the bounding box for the best prediction on the image from the first resize. 
        var outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage))
                                    .ConfigureAwait(false);

        return new ImageProcessingResult(outputImage, $"L:{eyeL}, R:{eyeR}");
    }

    List<FaceLandmarkPrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
    {
        // Setup inputs. Names used must match the input names in the model. 
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);
        var length = results.ToArray().Length;
        var confidences = results.ToArray()[length - 1].AsTensor<float>().ToArray();
        var points = new SKPoint[confidences.Length / 2];

        for (int i = 0, j = 0; i < (length = confidences.Length); i += 2)
        {
            points[j++] = new SKPoint(
                confidences[i + 0] * sourceImageWidth,
                confidences[i + 1] * sourceImageHeight);
        }

        return new List<FaceLandmarkPrediction>{
        new FaceLandmarkPrediction{
            Points = points
        }
    };
    }
}
