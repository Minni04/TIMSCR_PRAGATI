import 'package:flutter/material.dart';
import 'matching_products_page.dart'; // Ensure this is in the correct path

class ProductPage extends StatelessWidget {
  final String productName = "Awesome Product";
  final String productDescription =
      "Blue Kurti Pure Cotton";
  final double productPrice = 29.99;
  final String productImage = "assets/kurti.jpg"; // Update with your image

  final List<Map<String, String>> matchingProducts = [
    {
      "name": "Matching Product 1",
      "image": "assets/images/braclet.jpg",
      "price": "19.99"
    },
    {
      "name": "Matching Product 2",
      "image": "assets/images/shoe.jpg",
      "price": "24.99"
    },
    {
      "name": "Matching Product 3",
      "image": "assets/images/earing.jpg",
      "price": "34.99"
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Product Details"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    spreadRadius: 5,
                    blurRadius: 10,
                    offset: Offset(0, 3),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.asset('assets/images/kurti1.jpg', fit: BoxFit.cover),
              ),
            ),
            SizedBox(height: 16),
            Text(
              productName,
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 8),
            Text(
              "\$${productPrice.toStringAsFixed(2)}",
              style: TextStyle(fontSize: 20, color: Colors.green),
            ),
            SizedBox(height: 16),
            Text(
              productDescription,
              style: TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
            Spacer(),
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ElevatedButton(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text("Added to Cart!")),
                    );
                  },
                  child: Text("Add to Cart"),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 18),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => MatchingProductsPage(
                          matchingProducts: matchingProducts,
                        ),
                      ),
                    );
                  },
                  child: Text("Matching Pair"),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 15),
                    textStyle: TextStyle(fontSize: 18),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}
