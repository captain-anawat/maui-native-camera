using FaceDetectionLib.PrePostProcessing;

namespace FaceDetectionLib.OnnxModels.UltraFace;

public class UltrafacePrediction
{
    public PredictionBox Box { get; set; }
    public float Confidence { get; set; }
}