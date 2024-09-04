import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io' as io;
import 'package:dio/dio.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Supervisor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.red),
        useMaterial3: true,
        scaffoldBackgroundColor:   const Color.fromARGB(255, 191, 228, 204),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.brown, // AppBar color
          foregroundColor: Colors.white, // AppBar text color
        ),
      ),
      home: const MyHomePage(title: 'Supervisor'),
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
  final TextEditingController _numberController = TextEditingController();
  List<Map<String, dynamic>> _reports = [];
  List<String> _outputImages = [];
  String _statusMessage = '';

 Future<void> _requestPermissions() async {
  final status = await Permission.storage.request();
  if (!status.isGranted) {
    setState(() {
      _statusMessage = 'Storage permission denied';
    });
  } else {
    setState(() {
      _statusMessage = 'Storage permission granted';
    });
  }
}


  void _showContextMenu(BuildContext context, String imageUrl) {
    showMenu(
      context: context,
      position: RelativeRect.fromLTRB(0, 0, 0, 0),
      items: [
        PopupMenuItem(
          child: Text('Save Image'),
          value: 'save',
        ),
      ],
    ).then((value) {
      if (value == 'save') {
        _saveImage(imageUrl);
      }
    });
  }

 
Future<void> _saveImage(String imageUrl) async {
  try {
    final response = await Dio().get(
      imageUrl,
      options: Options(responseType: ResponseType.stream),
    );

    final directory = await getExternalStorageDirectory();
    final filePath = '${directory!.path}/${imageUrl.split('/').last}';

    final file = io.File(filePath);
    final raf = file.openSync(mode: io.FileMode.write);

    await response.data.stream.forEach((chunk) {
      raf.writeFromSync(chunk);
    });

    raf.closeSync();
    setState(() {
      _statusMessage = "Image saved to $filePath";
    });
  } catch (e) {
    setState(() {
      _statusMessage = "Error saving image: $e";
    });
  }
}

Future<io.Directory> _getSaveDirectory() async {
  if (io.Platform.isWindows) {
    return io.Directory(io.Directory.current.path); // Save in the current directory on Windows
  } else {
    return getApplicationDocumentsDirectory(); // Use Flutter's document directory for other platforms
  }
}

  Future<void> _sendNumberToBackend(int number) async {
    final url = Uri.parse('http://192.168.68.182:5000/set_number');
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: json.encode({"number": number}),
    );

    if (response.statusCode == 200) {
      setState(() {
        _outputImages = []; // Clear previous images
      });
      _fetchData();
    } else {
      setState(() {
        print(response.body);
        _statusMessage = 'Failed to send number. Status code: ${response.statusCode}';
      });
    }
  }

  Future<void> _fetchData() async {
    final url = Uri.parse('http://192.168.68.182:5000/data');

    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _outputImages = List<String>.from(data['images']);
          _reports = List<Map<String, dynamic>>.from(data['report']);
          print('Images: $_outputImages');
          print('Reports: $_reports');
          _statusMessage = 'Data loaded successfully';
        });
      } else {
        setState(() {
          _statusMessage = 'Failed to fetch data. Status code: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error fetching data: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Padding(
        padding: const EdgeInsets.all(6.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: [
                Container(
                  width: 200,
                  child: TextFormField(
                    controller: _numberController,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      labelText: 'Enter  number of ROI',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                ElevatedButton(
                  onPressed: () {
                    int number = int.tryParse(_numberController.text) ?? 0;
                    _sendNumberToBackend(number);
                  },
                  child: Text('Enter'),
                ),
              ],
            ),
            SizedBox(height: 20),
            Text(
              _statusMessage,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Text('Reports:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
           // Inside the ListView.builder for reports
Expanded(
  child: ListView.builder(
    itemCount: _reports.length,
    itemBuilder: (context, index) {
      final report = _reports[index];
      return Card(
        margin: EdgeInsets.symmetric(vertical: 8.0),
        color: Colors.brown[200], // Card color
        elevation: 5,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: report.entries.map((entry) {
              return RichText(
                text: TextSpan(
                  children: [
                    TextSpan(
                      text: '${entry.key}: ',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.brown[900], // Color for the key
                      ),
                    ),
                    TextSpan(
                      text: '${entry.value}',
                      style: TextStyle(
                        fontSize: 16,
                        fontStyle: FontStyle.italic,
                        color: Colors.brown[800], // Color for the value
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
      );
    },
  ),
),

            SizedBox(height: 20),
            Text('Processed Images:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            Expanded(
              child: ListView.separated(
                itemCount: _outputImages.length,
                separatorBuilder: (context, index) => SizedBox(height: 6.0),
                itemBuilder: (context, index) {
                  final imageUrl = 'http://192.168.68.182:5000/output/${_outputImages[index]}?t=${DateTime.now().millisecondsSinceEpoch}';
                  return GestureDetector(
                    onDoubleTap: () => _showContextMenu(context, imageUrl),
                    child: CachedNetworkImage(
                      imageUrl: imageUrl,
                      placeholder: (context, url) => CircularProgressIndicator(),
                      errorWidget: (context, url, error) => Icon(Icons.error),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
