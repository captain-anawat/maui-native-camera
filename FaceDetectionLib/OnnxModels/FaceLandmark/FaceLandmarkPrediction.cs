using SkiaSharp;

namespace FaceDetectionLib.OnnxModels.FaceLandmark;

internal class FaceLandmarkPrediction
{
    public SKPoint[] Points { get; set; }
}