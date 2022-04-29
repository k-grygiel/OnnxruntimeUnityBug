using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class SimpleOnnxTest : MonoBehaviour
{
    [SerializeField] private string neuralNetFilename;
    [SerializeField] private Text outputText;

    private string inputLayerName = "input_1";
    private Tensor<float> input;
    private InferenceSession sess;
    private List<NamedOnnxValue> inputs;

    void Start()
    {
        var modelPath = $"{Application.streamingAssetsPath}/{neuralNetFilename}";
        UnityEngine.Networking.UnityWebRequest www = UnityEngine.Networking.UnityWebRequest.Get(modelPath);
        www.SendWebRequest();
        while (!www.downloadHandler.isDone)
        {
        }

        SessionOptions opt = new SessionOptions();
        sess = new InferenceSession(www.downloadHandler.data, opt);

        input = new DenseTensor<float>(new[] { 1, 4000 });
        inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputLayerName, input)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = sess.Run(inputs);

        IEnumerable<float> output = results.First().AsEnumerable<float>();
        float sum = output.Sum(x => (float)Math.Exp(x));
        IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

        foreach (var item in softmax)
        {
            outputText.text += $"{item.ToString()}\n";
        }
    }
    
}
