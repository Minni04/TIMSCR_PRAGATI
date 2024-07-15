import 'package:flutter/material.dart';

class MatchingProductsPage extends StatelessWidget {
  final List<Map<String, String>> matchingProducts;

  MatchingProductsPage({required this.matchingProducts});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Matching Products"),
      ),
      body: ListView.builder(
        itemCount: matchingProducts.length,
        itemBuilder: (context, index) {
          final product = matchingProducts[index];
          return Card(
            margin: EdgeInsets.all(8),
            child: Container(
              height: 250, // Set a fixed height for each product card
              child: Row(
                children: [
                  // Display the product image
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Image.asset(
                      product["image"]!,
                      width: 100, // Set a fixed width for the image
                      fit: BoxFit.cover,
                    ),
                  ),
                  // Display the product details
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            product["name"]!,
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text(
                            "\$${product["price"]}",
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.green,
                            ),
                          ),
                          SizedBox(height: 16), // Space between text and buttons
                          Row(
                            children: [
                              ElevatedButton(
                                onPressed: () {
                                  // Add to cart logic
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text("${product['name']} added to cart!")),
                                  );
                                },
                                child: Text("Add to Cart"),
                                style: ElevatedButton.styleFrom(
                                  padding: EdgeInsets.symmetric(horizontal: 16),
                                  textStyle: TextStyle(fontSize: 14),
                                ),
                              ),
                              SizedBox(width: 8), // Space between buttons
                              ElevatedButton(
                                onPressed: () {
                                  // View details logic
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text("Viewing details for ${product['name']}")),
                                  );
                                },
                                child: Text("View Details"),
                                style: ElevatedButton.styleFrom(
                                  padding: EdgeInsets.symmetric(horizontal: 16),
                                  textStyle: TextStyle(fontSize: 14),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
