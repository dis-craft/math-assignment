import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import scipy.integrate as spi
from fpdf import FPDF
import io
import os

# Disclaimer
st.markdown("""
**DISCLAIMER:**  
This tool is provided for educational purposes only. The developer of this tool is not responsible for any sort of consequences that arise by using this tool. By using this tool, you agree 
that you are solely responsible for any outcomes resulting from its use.
""")

agree = st.checkbox("I agree to the terms and conditions above")
if not agree:
    st.stop()

# PDF Generation Function
def create_pdf(name, usn, section, experiment, graph_img, code, results):
    class PDF(FPDF):
        def header(self):
            # Header with standard font
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, 'Mathematics Experiment Report', ln=True, align='C')
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # Student Info
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Name: {name}', ln=True)
    pdf.cell(0, 10, f'USN: {usn}', ln=True)
    pdf.cell(0, 10, f'Section: {section}', ln=True)
    pdf.ln(15)
    
    # Experiment Title
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, experiment, ln=True, align='C')
    pdf.ln(10)
    
    # Results
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Results:', ln=True)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 8, results)
    pdf.ln(10)
    
    # Graph
    if graph_img:
        img_path = "graph.png"
        with open(img_path, "wb") as f:
            f.write(graph_img.getbuffer())
        pdf.image(img_path, x=10, w=190)
        pdf.ln(15)
    
    # Code
    pdf.set_font('Courier', 'B', 12)
    pdf.cell(0, 10, 'Experiment Code:', ln=True)
    pdf.set_font('Courier', '', 10)
    pdf.multi_cell(0, 5, code)
    
    pdf.output("report.pdf")
    return "report.pdf"

# Streamlit UI
st.title("Mathematics Experiment Simulator")

# Student Details
name = st.text_input("Full Name")
usn = st.text_input("USN (University Seat Number)")
section = st.text_input("Section")

experiment = st.selectbox("Select Experiment", [
    "Projectile Motion Analysis",
    "Rainfall Probability Estimation",
    "3D Hit Probability Distribution"
])

if experiment == "Projectile Motion Analysis":
    st.header("Projectile Motion Trajectories")
    
    if st.button("Generate Report"):
        # Generate plot and results
        def projectile_motion(v0, theta, g=9.81):
            theta_rad = np.radians(theta)
            t_flight = 2 * v0 * np.sin(theta_rad) / g
            t = np.linspace(0, t_flight, 100)
            x = v0 * np.cos(theta_rad) * t
            y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
            return x, y, t_flight

        angles = [30, 45, 60]
        v0 = 20
        results = []
        g = 9.81

        fig = plt.figure(figsize=(10, 5))
        for angle in angles:
            x, y, t_flight = projectile_motion(v0, angle)
            plt.plot(x, y, label=f"{angle}°")
            max_height = (v0**2 * np.sin(np.radians(angle))**2)/(2*g)
            range_val = (v0**2 * np.sin(2*np.radians(angle)))/g
            results.append(
                f"Angle {angle}°:\n"
                f"- Time of Flight: {t_flight:.2f} s\n"
                f"- Max Height: {max_height:.2f} m\n"
                f"- Range: {range_val:.2f} m\n"
            )

        plt.xlabel("Horizontal Distance (m)")
        plt.ylabel("Vertical Distance (m)")
        plt.title("Projectile Motion for Different Angles")
        plt.legend()
        plt.grid()
        
        # Save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Format results
        results_text = "\n".join(results)
        
        # PDF Code
        code = '''import numpy as np
import matplotlib.pyplot as plt

def projectile_motion(v0, theta, g=9.81):
    theta_rad = np.radians(theta)
    t_flight = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_flight, 100)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return x, y

angles = [30, 45, 60]
v0 = 20

plt.figure(figsize=(10, 5))
for angle in angles:
    x, y = projectile_motion(v0, angle)
    plt.plot(x, y, label=f"{angle}°")
    
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.title("Projectile Motion for Different Angles")
plt.legend()
plt.grid()
plt.show()'''
        
        # Generate PDF
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="projectile_report.pdf")

elif experiment == "Rainfall Probability Estimation":
    st.header("Rainfall Probability Analysis")
    
    if st.button("Generate Report"):
        # Generate plot and results
        mu, sigma = 50, 15
        pdf_func = lambda x: norm.pdf(x, mu, sigma)
        prob, _ = spi.quad(pdf_func, 30, 70)

        fig = plt.figure(figsize=(8, 5))
        x = np.linspace(0, 100, 100)
        y = pdf_func(x)
        plt.plot(x, y, label="Rainfall Distribution")
        plt.fill_between(x, y, where=(x >= 30) & (x <= 70), color="green", alpha=0.5)
        plt.xlabel("Rainfall (mm)")
        plt.ylabel("Probability Density")
        plt.title(f"Probability of Rainfall between 30mm and 70mm: {prob:.2f}")
        plt.legend()
        plt.grid()
        
        # Save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Format results (using text instead of Greek characters)
        results_text = f"Calculated Probability: {prob:.4f}\n\n"
        results_text += f"Normal Distribution Parameters:\n- Mean (mu) = {mu}\n- Std Dev (sigma) = {sigma}"
        
        # PDF Code
        code = '''import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 50, 15
pdf = lambda x: norm.pdf(x, mu, sigma)
prob, _ = spi.quad(pdf, 30, 70)

x = np.linspace(0, 100, 100)
y = pdf(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Rainfall Distribution")
plt.fill_between(x, y, where=(x >= 30) & (x <= 70), color="green", alpha=0.5)
plt.xlabel("Rainfall (mm)")
plt.ylabel("Probability Density")
plt.title(f"Probability between 30mm-70mm: {prob:.2f}")
plt.legend()
plt.grid()
plt.show()'''
        
        # Generate PDF
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="rainfall_report.pdf")

elif experiment == "3D Hit Probability Distribution":
    st.header("3D Hit Probability Analysis")
    
    if st.button("Generate Report"):
        # Generate plot and results
        mean = [10, 20, 15]
        cov = [[4, 1, 1], [1, 3, 1], [1, 1, 2]]
        x, y, z = np.random.multivariate_normal(mean, cov, 1000).T

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, alpha=0.5)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Projectile Hit Probability Distribution")
        
        # Save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Format results
        results_text = f"Mean Vector:\n{mean}\n\n"
        results_text += "Covariance Matrix:\n"
        for row in cov:
            results_text += f"{row}\n"
        
        # PDF Code
        code = '''import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean = [10, 20, 15]
cov = [[4, 1, 1], [1, 3, 1], [1, 1, 2]]
x, y, z = np.random.multivariate_normal(mean, cov, 1000).T

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, alpha=0.5)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("Projectile Hit Probability Distribution")
plt.show()'''
        
        # Generate PDF
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="3d_probability_report.pdf")