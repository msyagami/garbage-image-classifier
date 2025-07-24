import 'package:flutter/material.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.lightBlue),
      ),
      home: const MyHomePage(title: 'Garbage Classifier'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  void _navigateToClassifier() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const GarbageClassifierScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: _navigateToClassifier,
              child: const Text('Classify Garbage Image'),
            ),
          ],
        ),
      ),
    );
  }
}

class GarbageClassifierScreen extends StatefulWidget {
  const GarbageClassifierScreen({super.key});

  @override
  State<GarbageClassifierScreen> createState() =>
      _GarbageClassifierScreenState();
}

class _GarbageClassifierScreenState extends State<GarbageClassifierScreen> {
  dynamic _model;
  String? _result;
  File? _image;
  bool _loading = false;
  bool _modelLoaded = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    _model = await PytorchLite.loadClassificationModel(
      'assets/models/best_garbage_classifier_v9_lite.pt',
      128,
      128,
      10, // Adjust if needed
      labelPath: 'assets/label_classification.txt',
    );
    setState(() {
      _modelLoaded = true;
    });
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = null;
      });
      _classifyImage();
    }
  }

  Future<void> _classifyImage() async {
    if (_image == null || _model == null) return;
    setState(() {
      _loading = true;
    });
    try {
      final bytes = await _image!.readAsBytes();
      // Get raw scores for all classes
      final List<double> scores = await _model.getImagePredictionList(bytes);
      // Get probabilities using softmax
      final List<double> probabilities = _model.getProbabilities(scores);
      // Find the index of the highest probability
      int maxIdx = 0;
      double maxProb = 0.0;
      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIdx = i;
        }
      }
      // Get label from model labels
      String labelStr =
          _model.labels.isNotEmpty && maxIdx < _model.labels.length
          ? _model.labels[maxIdx]
          : 'Unknown';
      final confidencePercent = (maxProb * 100).toStringAsFixed(2);
      debugPrint('Scores: $scores');
      debugPrint('Probabilities: $probabilities');
      debugPrint('Predicted label: $labelStr, Confidence: $confidencePercent%');
      _result = 'Label: $labelStr\nConfidence: $confidencePercent%';
    } catch (e) {
      _result = 'Error: ${e.toString()}';
    }
    setState(() {
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Garbage Classifier')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: <Widget>[
            ElevatedButton(
              onPressed: _modelLoaded ? _pickImage : null,
              child: Text(_modelLoaded ? 'Pick Image' : 'Loading Model...'),
            ),
            if (_image != null) ...[
              const SizedBox(height: 16),
              Image.file(_image!, height: 200),
            ],
            if (_loading) ...[
              const SizedBox(height: 16),
              const CircularProgressIndicator(),
            ],
            if (_result != null) ...[
              const SizedBox(height: 16),
              Text(_result!),
            ],
          ],
        ),
      ),
    );
  }
}

// ...existing code...
