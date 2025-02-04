// Prompt for Creating a Mathematical Experiment Report Generator Web App

// Objective:
// Develop a full-stack web application that allows users to:

// Select from three pre-built Python-based mathematical experiments (Projectile Motion, Rainfall Prediction, 3D Target Probability).

// Generate downloadable reports (PDF/DOCX) containing their details, the experiment's Python code, and outputs.

// Log all user activity to Firebase.

// Key Requirements:

// Frontend:

// Modern, responsive UI with a clean design (gradients, animations, form validation).
// INput field: it should ask with a radio button whether they want the names printed on it or no. but we still collect in the firebase. 

// Input fields: Name, USN, Section.

// Dropdown for experiment selection and format toggle (PDF/DOCX).

// Download button triggering report generation.

// Backend & Integrations:

// Firebase Firestore: Use the provided config to log user details, experiment choice, format, and timestamp.
// collect everybit of information u can about the user. make sure to search all the web to what is possibly can be collected about the user.

// Document Generation:

// PDF: Use jsPDF with code formatted in monospace font.

// DOCX: Use docx.js with styled code blocks and headings.

// Python Code Embedding: Include actual experiment code in reports (see examples below).

// Pre-Written Python Experiments:

// Projectile Motion: Simulate trajectory with outputs for max height/range.

// Rainfall Prediction: Calculate probability using integration.

// 3D Target Probability: Estimate hit chance with triple integration.

// const experiments = {
//             1: {
//                 code: `import numpy as np
// import matplotlib.pyplot as plt

// def projectile_motion(v0, theta):
//     g = 9.81
//     theta = np.radians(theta)
//     t_flight = 2 * v0 * np.sin(theta) / g
//     t = np.linspace(0, t_flight, 100)
//     x = v0 * np.cos(theta) * t
//     y = v0 * np.sin(theta) * t - 0.5 * g * t**2
//     return x, y

// # Example usage
// v0 = 50  # m/s
// theta = 45  # degrees
// x, y = projectile_motion(v0, theta)
// print(f"Max height: {max(y):.2f} m")
// print(f"Range: {max(x):.2f} m")`,
//                 output: "Max height: 63.77 m\nRange: 254.84 m"
//             },
//             2: {
//                 code: `from scipy import integrate
// import numpy as np

// def rainfall_probability(pdf, a, b):
//     result, _ = integrate.quad(pdf, a, b)
//     return result

// # Example PDF (normal distribution)
// pdf = lambda x: np.exp(-(x-50)**2/(2*10**2)) / (10*np.sqrt(2*np.pi))
// a, b = 45, 55
// prob = rainfall_probability(pdf, a, b)
// print(f"Probability of rainfall between {a}-{b} mm: {prob*100:.1f}%")`,
//                 output: "Probability of rainfall between 45-55 mm: 38.3%"
//             },
//             3: {
//                 code: `import sympy as sp

// def hit_probability(a, b, c, f):
//     x, y, z = sp.symbols('x y z')
//     integral = sp.integrate(f, (x, -a, a), (y, -b, b), (z, -c, c))
//     return integral

// # Example: Cubic target area with normal distribution
// a, b, c = 2, 2, 2
// f = sp.exp(-(x**2 + y**2 + z**2))
// prob = hit_probability(a, b, c, f)
// print(f"Hit probability: {prob.evalf()*100:.2f}%")`,
//                 output: "Hit probability: 78.12%"
//             }
//         };


// Technical Specs:

// Use plain JavaScript (no frameworks).

// Ensure mobile-friendly design.

// Include loading states and error handling.

// Example Python Code Snippets (to include in reports):

// Firebase Config (to use):

// javascript
// Copy
// const firebaseConfig = {
//             apiKey: "AIzaSyBSbonwVE3PPXIIrSrvrB75u2AQ_B_Tni4",
//             authDomain: "discraft-c1c41.firebaseapp.com",
//             databaseURL: "https://discraft-c1c41-default-rtdb.firebaseio.com",
//             projectId: "discraft-c1c41",
//             storageBucket: "discraft-c1c41.appspot.com",
//             messagingSenderId: "525620150766",
//             appId: "1:525620150766:web:a426e68d206c68764aceff",
//             measurementId: "G-2TRNRYRX5E"
//         };  
// Deliverables:

// Single HTML file with embedded CSS/JS.

// Working document generation for both formats.

// Firebase realtime database integration for logging.

// Well-structured, commented code.