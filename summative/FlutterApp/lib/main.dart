import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const CalmPulseApp());
}

class CalmPulseApp extends StatelessWidget {
  const CalmPulseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CalmPulse Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _price = TextEditingController(text: '0');
  final _ratingCount = TextEditingController(text: '3500');
  final _sizeMb = TextEditingController(text: '110');
  final _languageCount = TextEditingController(text: '12');
  final _ageDays = TextEditingController(text: '2200');
  final _updateRecency = TextEditingController(text: '30');
  final _minIos = TextEditingController(text: '13.0');

  bool _hasIap = true;
  bool _hasSupport = true;
  bool _isGameCenter = false;
  String _genre = 'Health & Fitness';
  String _contentRating = '4+';

  String _result = '';
  bool _loading = false;

  static const List<String> genres = [
    'Book',
    'Business',
    'Education',
    'Entertainment',
    'Finance',
    'Food & Drink',
    'Games',
    'Graphics & Design',
    'Health & Fitness',
    'Lifestyle',
    'Medical',
    'Music',
    'News',
    'Photo & Video',
    'Productivity',
    'Reference',
    'Shopping',
    'Utilities',
  ];

  static const List<String> contentRatings = ['4+', '9+', '12+', '17+'];

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    final payload = {
      'price': double.parse(_price.text),
      'rating_count': int.parse(_ratingCount.text),
      'size_mb': double.parse(_sizeMb.text),
      'primary_genre': _genre,
      'content_rating': _contentRating,
      'language_count': int.parse(_languageCount.text),
      'has_iap': _hasIap,
      'has_support_url': _hasSupport,
      'min_ios': _minIos.text,
      'is_game_center': _isGameCenter,
      'age_days': int.parse(_ageDays.text),
      'update_recency_days': int.parse(_updateRecency.text),
    };

    setState(() {
      _loading = true;
      _result = '';
    });

    const baseUrl = String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'https://your-render-app.onrender.com',
    );

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _result = 'Predicted rating: ${data['predicted_rating']}\n'
              'Model: ${data['model_name']} (RMSE ${data['test_rmse']})';
        });
      } else {
        setState(() {
          _result = 'Error: ${response.statusCode} ${response.body}';
        });
      }
    } catch (err) {
      setState(() {
        _result = 'Network error: $err';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _price.dispose();
    _ratingCount.dispose();
    _sizeMb.dispose();
    _languageCount.dispose();
    _ageDays.dispose();
    _updateRecency.dispose();
    _minIos.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CalmPulse Rating Predictor'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildNumberField('Price (USD)', _price, min: 0, max: 100, isDouble: true),
              _buildNumberField('Rating Count', _ratingCount, min: 1, max: 4000000),
              _buildNumberField('Size (MB)', _sizeMb, min: 1, max: 2000, isDouble: true),
              _buildDropdown('Primary Genre', _genre, genres, (value) {
                setState(() => _genre = value!);
              }),
              _buildDropdown('Content Rating', _contentRating, contentRatings, (value) {
                setState(() => _contentRating = value!);
              }),
              _buildNumberField('Language Count', _languageCount, min: 1, max: 60),
              _buildSwitch('Has In-App Purchases', _hasIap, (value) {
                setState(() => _hasIap = value);
              }),
              _buildSwitch('Has Support URL', _hasSupport, (value) {
                setState(() => _hasSupport = value);
              }),
              TextFormField(
                controller: _minIos,
                decoration: const InputDecoration(labelText: 'Minimum iOS (e.g., 13.0)'),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Enter minimum iOS version';
                  }
                  if (!RegExp(r'^\d{1,2}(\.\d)?$').hasMatch(value)) {
                    return 'Format must be like 13.0';
                  }
                  return null;
                },
              ),
              _buildSwitch('Supports Game Center', _isGameCenter, (value) {
                setState(() => _isGameCenter = value);
              }),
              _buildNumberField('Age (days)', _ageDays, min: 0, max: 7000),
              _buildNumberField('Update recency (days)', _updateRecency, min: 0, max: 5000),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _loading ? null : _predict,
                  child: _loading
                      ? const CircularProgressIndicator.adaptive()
                      : const Text('Predict'),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                _result,
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNumberField(
    String label,
    TextEditingController controller, {
    required int min,
    required int max,
    bool isDouble = false,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(labelText: label),
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Enter $label';
        }
        final parsed = isDouble ? double.tryParse(value) : int.tryParse(value);
        if (parsed == null) {
          return 'Invalid number';
        }
        final numeric = parsed.toDouble();
        if (numeric < min) {
          return 'Must be >= $min';
        }
        if (numeric > max) {
          return 'Must be <= $max';
        }
        return null;
      },
    );
  }

  Widget _buildDropdown(
    String label,
    String value,
    List<String> options,
    ValueChanged<String?> onChanged,
  ) {
    return DropdownButtonFormField<String>(
      value: value,
      decoration: InputDecoration(labelText: label),
      items: options
          .map((option) => DropdownMenuItem(
                value: option,
                child: Text(option),
              ))
          .toList(),
      onChanged: onChanged,
    );
  }

  Widget _buildSwitch(String label, bool value, ValueChanged<bool> onChanged) {
    return SwitchListTile(
      title: Text(label),
      value: value,
      onChanged: onChanged,
    );
  }
}
