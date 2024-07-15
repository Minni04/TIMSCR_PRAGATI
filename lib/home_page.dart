import 'package:flutter/material.dart';
import 'product_page.dart'; // Import Product Page
import 'image_upload_page.dart'; // Import Image Upload Page

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Home'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Add your image here
            Image.asset(
              'assets/images/home.png', // Replace with your image path
              height: 250, // Set the desired height
            ),
            SizedBox(height: 20),
            Text(
              'Welcome to Myntra',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 40),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => ProductPage()),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink, // Set button color to pink
                foregroundColor: Colors.white, // Set text color to white
              ),
              child: Text('Products'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => ImageUploadPage()),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink, // Set button color to pink
                foregroundColor: Colors.white, // Set text color to white
              ),
              child: Text('Do Your Personal Color Analysis'),
            ),
          ],
        ),
      ),
    );
  }
}
