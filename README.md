# Pathfinding Algorithms in Rust: A* and Dijkstra's

This repository contains the implementation of two prominent pathfinding algorithms, **A*** and **Dijkstra's**, in Rust. The project explores both serialized and parallelized versions of these algorithms, leveraging Rust's concurrency features to improve performance on large-scale graphs.

---

## **Features**

- **A***:
  - Serialized implementation.
  - Parallelized implementation using threads and synchronization mechanisms.
  - Incorporates a heuristic function for faster convergence.

- **Dijkstra's**:
  - Serialized implementation.
  - Parallelized implementation optimized for large graphs.

- **Input Handling**:
  - Reads adjacency and weight matrices from input files.
  - Supports graphs of varying sizes, tested on inputs of up to 10,000 nodes.

- **Results and Performance**:
  - Comprehensive benchmarking of serialized vs. parallel versions.
  - Demonstrates speedup and scalability for large graphs.

---

## **Usage**

### **Prerequisites**
- Install [Rust](https://www.rust-lang.org/tools/install).
- Ensure the input graph files are formatted correctly (see the **Input Format** section below).

### **Running the Code**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
