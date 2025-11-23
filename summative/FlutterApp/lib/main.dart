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
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF5F7FB),
        useMaterial3: true,
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
            textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFCBD5F5)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFCBD5F5)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFF2563EB), width: 1.5),
          ),
          labelStyle: const TextStyle(color: Color(0xFF475467)),
        ),
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
      defaultValue: 'https://linear-regression-model-mxox.onrender.com',
    );
    final sanitizedBase = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    try {
      final response = await http.post(
        Uri.parse('$sanitizedBase/predict'),
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
        foregroundColor: const Color(0xFF0F172A),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Forecast wellbeing ratings',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF0F172A),
                    ),
              ),
              const SizedBox(height: 6),
              const Text(
                'Feed CalmPulse signals from store telemetry to spot UX wins before launch.',
                style: TextStyle(color: Color(0xFF475467)),
              ),
              const SizedBox(height: 20),
              Card(
                elevation: 0,
                color: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildNumberField('Price (USD)', _price, min: 0, max: 100, isDouble: true),
                        const SizedBox(height: 12),
                        _buildNumberField('Rating Count', _ratingCount, min: 1, max: 4000000),
                        const SizedBox(height: 12),
                        _buildNumberField('Size (MB)', _sizeMb, min: 1, max: 2000, isDouble: true),
                        const SizedBox(height: 12),
                        _buildDropdown('Primary Genre', _genre, genres, (value) {
                          setState(() => _genre = value!);
                        }),
                        const SizedBox(height: 12),
                        _buildDropdown('Content Rating', _contentRating, contentRatings, (value) {
                          setState(() => _contentRating = value!);
                        }),
                        const SizedBox(height: 12),
                        _buildNumberField('Language Count', _languageCount, min: 1, max: 60),
                        const SizedBox(height: 6),
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
                        const SizedBox(height: 12),
                        _buildNumberField('Update recency (days)', _updateRecency, min: 0, max: 5000),
                        const SizedBox(height: 18),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: _loading ? null : _predict,
                            child: _loading
                                ? const CircularProgressIndicator.adaptive()
                                : const Text('Predict rating'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              AnimatedOpacity(
                opacity: _result.isEmpty ? 0 : 1,
                duration: const Duration(milliseconds: 250),
                child: _result.isEmpty
                    ? const SizedBox.shrink()
                    : Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF0F172A),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Text(
                          _result,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
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
      contentPadding: EdgeInsets.zero,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      tileColor: const Color(0xFFF2F4F7),
      title: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
      value: value,
      onChanged: onChanged,
    );
  }
}
