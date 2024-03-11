using FaceDetectionLib.PrePostProcessing;

namespace FaceDetectionLib;

public interface IVisionSample
{
    string Name { get; }
    string ModelName { get; }
    Task InitializeAsync();
    Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider);
    Task<ImageProcessingResult> ProcessImageAsync(byte[] image);
    string GetStringData();
}
