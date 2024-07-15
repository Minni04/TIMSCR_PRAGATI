// results_screen.dart

import 'package:flutter/material.dart';

class ResultsScreen extends StatelessWidget {
  final String result;

  ResultsScreen({required this.result});

  @override
  Widget build(BuildContext context) {
    Color myColor = Color(0xffc85a32);
    Color myColor2 = Color(0xff6b7c53);
    Color myColor3 = Color(0xfffdd835);
    Color myColor4 = Color(0xff800020);
    Color myColor5 = Color(0xffe27c54);

    return Scaffold(
      appBar: AppBar(title: Text('Results')),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Center(
          child: Column(
            children: <Widget>[
              Text("AUTUMN", style: TextStyle(fontSize: 24)),
              Text("MATCHING PAIRS", style: TextStyle(fontSize: 20)),
              Row(
                children: [
                  Expanded(
                    child: Container(
                      color: myColor2,
                      child: Column(children: [Text("Olive Green")]),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      color: myColor3,
                      child: Column(children: [Text("Golden Yellow")]),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      color: myColor,
                      child: Column(children: [Text("Burnt Orange")]),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),
              Text("BEST LIP COLOUR", style: TextStyle(fontSize: 20)),
              Row(
                children: [
                  Expanded(
                    child: Container(
                      color: myColor4,
                      child: Column(children: [Text("Brick Red")]),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      color: myColor5,
                      child: Column(children: [Text("Terracotta")]),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),
              Text(result),
            ],
          ),
        ),
      ),
    );
  }
}
