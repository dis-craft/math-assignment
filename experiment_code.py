import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as spi
from fpdf import FPDF
import io
import os

# ----------------------- Disclaimer -----------------------
st.markdown("""
**DISCLAIMER:**  
This tool is provided for educational purposes only. The developer of this tool is not responsible for any consequences arising from its use. By using this tool, you agree 
that you are solely responsible for any outcomes resulting from its use.
""")

# Checkbox to agree to the disclaimer; if not checked, stop execution.
agree = st.checkbox("I agree to the terms and conditions above")
if not agree:
    st.stop()

# ----------------------- Input Section -----------------------
st.title("Mathematics Experiment Simulator")

# Student details
name = st.text_input("Full Name")
usn = st.text_input("USN (University Seat Number)")
section = st.text_input("Section")

# ✅ New checkbox for including Name, USN, and Section in the PDF
include_name = st.checkbox("Include my details (Name, USN, Section) in the PDF?", value=True)

# Experiment selection
experiment = st.selectbox("Select Experiment", [
    "Projectile Motion Analysis",
    "Rainfall Probability Estimation",
    "3D Hit Probability Distribution"
])

# ----------------------- Fancy PDF Generation Function -----------------------
def create_pdf(name, usn, section, experiment, graimport streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as spi
from fpdf import FPDF
import io
import os
import base64
import random

# ----------------------- Custom CSS for Styling -----------------------
st.markdown("""
    <style>
    /* Style for centered, larger buttons */
    div.stButton > button {
        width: 220px;
        height: 50px;
        font-size: 18px;
        margin: 10px auto;
        display: block;
    }
    /* Center the demo PDF preview container */
    .pdf-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Disclaimer -----------------------
st.markdown("""
**DISCLAIMER:**  
This tool is provided for educational purposes only. The developer of this tool is not responsible for any consequences arising from its use. By using this tool, you agree 
that you are solely responsible for any outcomes resulting from its use.
""")

agree = st.checkbox("I agree to the above terms and conditions")
if not agree:
    st.stop()

# ----------------------- Input Section -----------------------
st.title("Mathematics Experiment Simulator")

with st.container():
    st.subheader("User Details")
    name = st.text_input("Full Name")
    usn = st.text_input("USN (University Seat Number)")
    section = st.text_input("Section")
    include_name = st.checkbox("Include my details (Name, USN, Section) in the PDF?", value=True)

# ----------------------- Experiment Selection -----------------------
with st.container():
    st.subheader("Experiment Selection")
    experiment = st.selectbox("Select Experiment", [
        "Projectile Motion Analysis",
        "Rainfall Probability Estimation",
        "3D Hit Probability Distribution"
    ])

# ----------------------- Design Selection -----------------------
with st.container():
    st.subheader("Select PDF Layout Design")
    # Define friendly design names and map them to internal design IDs.
    design_dict = {
        "Elegant Teal": "Design 1",
        "Modern Gray": "Design 2",
        "Minimalist Underline": "Design 3",
        "Navy Formal": "Design 4",
        "Pastel Contemporary": "Design 5",
        "Vibrant Pink": "Design 6"
    }
    design_choice = st.selectbox("Choose a design", list(design_dict.keys()))
    
    # Optionally display a preview image for the selected design.
    preview_image_path = f"{design_choice.lower().replace(' ', '_')}_preview.png"
    if os.path.exists(preview_image_path):
        st.image(preview_image_path, caption=f"{design_choice} Preview", use_column_width=True)
    else:
        st.info(f"Preview image for {design_choice} not available. Add an image named '{preview_image_path}' to see a preview.")

# ----------------------- Demo PDF Preview -----------------------
with st.container():
    st.subheader("Demo PDF Preview")
    if st.button("Preview Demo PDF"):
        # Generate a demo PDF using sample/demo data.
        demo_name = "Demo User"
        demo_usn = "DemoUSN"
        demo_section = "A"
        demo_experiment = "Demo Experiment: Sample demonstration of the PDF design."
        demo_code = "print('Hello, World!')\n# This is a demo code snippet."
        demo_results = "These are demo results. Replace this text with your experiment results."
        
        # Create a simple dummy plot for the demo graph.
        fig = plt.figure(figsize=(6, 3))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y, label="Demo Plot")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        buf_demo = io.BytesIO()
        plt.savefig(buf_demo, format='png')
        buf_demo.seek(0)
        plt.close(fig)
        
        # Use the selected design.
        design_id = design_dict[design_choice]
        demo_pdf_path = "demo_report.pdf"
        
        # Call the PDF creation function (defined below) with demo data.
        def create_pdf(name, usn, section, experiment, graph_img, code, results, include_name, layout_design):
            """
            Creates a styled PDF report using one of six layout designs.
            """
            if layout_design == "Design 1":
                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(0, 128, 128)
                        self.set_text_color(255, 255, 255)
                        self.set_font('Arial', 'B', 20)
                        self.cell(0, 15, "MATHEMATICS INNOVATIVE EXPERIMENT", ln=1, align='C', fill=True)
                        self.ln(3)
                        self.set_line_width(1.5)
                        self.set_draw_color(0, 0, 0)
                        self.rect(5, 5, self.w - 10, self.h - 10)
                        self.set_text_color(0, 0, 0)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 10)
                        self.set_text_color(100, 100, 100)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            elif layout_design == "Design 2":
                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(80, 80, 80)
                        self.set_text_color(255, 255, 255)
                        self.set_font('Helvetica', 'B', 22)
                        self.cell(0, 15, "Mathematics Experiment", ln=1, align='C', fill=True)
                        self.ln(2)
                        self.set_line_width(1)
                        self.set_draw_color(80, 80, 80)
                        self.rect(5, 5, self.w - 10, self.h - 10)
                        self.set_text_color(0, 0, 0)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Helvetica', 'I', 10)
                        self.set_text_color(120, 120, 120)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            elif layout_design == "Design 3":
                class PDF(FPDF):
                    def header(self):
                        self.set_font('Times', 'B', 24)
                        self.cell(0, 15, "Mathematics Experiment", ln=1, align='C')
                        self.ln(2)
                        self.set_line_width(0.5)
                        self.line(10, 25, self.w - 10, 25)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Times', 'I', 10)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            elif layout_design == "Design 4":
                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(0, 0, 128)
                        self.set_text_color(255, 255, 255)
                        self.set_font('Courier', 'B', 20)
                        self.cell(0, 15, "Mathematics Experiment Report", ln=1, align='C', fill=True)
                        self.ln(3)
                        self.set_line_width(1.5)
                        self.set_draw_color(0, 0, 128)
                        self.rect(5, 5, self.w - 10, self.h - 10)
                        self.set_text_color(0, 0, 0)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Courier', 'I', 10)
                        self.set_text_color(100, 100, 100)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            elif layout_design == "Design 5":
                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(173, 216, 230)
                        self.set_text_color(0, 0, 128)
                        self.set_font('Arial', 'B', 22)
                        self.cell(0, 15, "Mathematics Experiment", ln=1, align='C', fill=True)
                        self.ln(3)
                        self.set_line_width(1)
                        self.set_draw_color(173, 216, 230)
                        self.rect(5, 5, self.w - 10, self.h - 10)
                        self.set_text_color(0, 0, 0)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 10)
                        self.set_text_color(120, 120, 120)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            elif layout_design == "Design 6":
                class PDF(FPDF):
                    def header(self):
                        self.set_fill_color(255, 105, 180)
                        self.set_text_color(255, 255, 255)
                        self.set_font('Arial', 'B', 20)
                        self.cell(0, 15, "Innovative Mathematics Experiment", ln=1, align='C', fill=True)
                        self.ln(3)
                        self.set_line_width(2)
                        self.set_draw_color(255, 105, 180)
                        self.rect(5, 5, self.w - 10, self.h - 10)
                        self.set_text_color(0, 0, 0)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 10)
                        self.set_text_color(150, 150, 150)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            pdf = PDF()
            pdf.add_page()
            
            # QUESTION Section
            pdf.set_fill_color(200, 230, 201)
            pdf.set_text_color(34, 139, 34)
            pdf.set_font('Times', 'B', 16)
            pdf.cell(0, 12, "QUESTION:", ln=True, fill=True)
            pdf.ln(2)
            pdf.set_font('Times', '', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 8, experiment)
            pdf.ln(4)
            
            # Student Information Section
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(230, 230, 250)
            if include_name:
                pdf.cell(0, 10, f"NAME: {name}", ln=True, align="C", fill=True)
                pdf.cell(0, 10, f"USN: {usn}", ln=True, align="C", fill=True)
                pdf.cell(0, 10, f"SECTION: {section}", ln=True, align="C", fill=True)
                pdf.ln(4)
            
            # EXPERIMENT CODE Section
            pdf.set_fill_color(255, 228, 196)
            pdf.set_font('Courier', 'B', 18)
            pdf.cell(0, 10, "EXPERIMENT CODE:", ln=True, fill=True)
            pdf.set_font('Courier', '', 10)
            pdf.multi_cell(0, 5, code)
            pdf.ln(4)
            
            # OUTPUT WITH GRAPH Section
            pdf.set_fill_color(255, 240, 245)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "OUTPUT WITH GRAPH:", ln=True, fill=True)
            pdf.ln(2)
            if graph_img:
                img_path = "graph.png"
                with open(img_path, "wb") as f:
                    f.write(graph_img.getbuffer())
                pdf.image(img_path, x=20, w=pdf.w - 40)
                pdf.ln(5)
                os.remove(img_path)
            
            # RESULTS Section
            pdf.set_fill_color(255, 250, 205)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "RESULTS:", ln=True, fill=True)
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 8, results)
            
            pdf_file = "demo_report.pdf"
            pdf.output(pdf_file)
            return pdf_file
        
        demo_pdf = create_pdf(demo_name, demo_usn, demo_section, demo_experiment,
                              buf_demo, demo_code, demo_results, True, design_id)
        
        # Display the demo PDF in a read-only iframe (no controls)
        with open(demo_pdf, "rb") as f:
            demo_base64 = base64.b64encode(f.read()).decode('utf-8')
        demo_iframe = f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{demo_base64}" width="700" height="900" style="pointer-events: none;" type="application/pdf"></iframe></div>'
        st.markdown(demo_iframe, unsafe_allow_html=True)

# ----------------------- Experiment-Specific PDF Generation -----------------------
st.markdown("---")
st.subheader("Generate Your Experiment Report")

if experiment == "Projectile Motion Analysis":
    st.header("Projectile Motion Trajectories")
    if st.button("Generate and Preview Report"):
        # Generate sample data for projectile motion.
        v0 = np.random.randint(10, 31)
        angles = np.random.randint(20, 71, size=3)
        g = 9.81

        def projectile_motion(v0, theta, g=9.81):
            theta_rad = np.radians(theta)
            t_flight = 2 * v0 * np.sin(theta_rad) / g
            t = np.linspace(0, t_flight, 100)
            x = v0 * np.cos(theta_rad) * t
            y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
            return x, y, t_flight

        fig = plt.figure(figsize=(10, 5))
        results_lines = []
        for theta in angles:
            x, y, t_flight = projectile_motion(v0, theta)
            plt.plot(x, y, label=f"{theta}°")
            max_height = (v0**2 * np.sin(np.radians(theta))**2) / (2 * g)
            range_val = (v0**2 * np.sin(2 * np.radians(theta))) / g
            results_lines.append(
                f"Angle {theta}°: Time of Flight: {t_flight:.2f} s | Max Height: {max_height:.2f} m | Range: {range_val:.2f} m"
            )
        plt.xlabel("Horizontal Distance (m)")
        plt.ylabel("Vertical Distance (m)")
        plt.title(f"Projectile Motion (v0 = {v0} m/s)")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        results_text = f"Initial Speed (v0): {v0} m/s\n" + "\n".join(results_lines)
        code = f'''import numpy as np
import matplotlib.pyplot as plt

def projectile_motion(v0, theta, g=9.81):
    theta_rad = np.radians(theta)
    t_flight = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_flight, 100)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return x, y, t_flight

v0 = {v0}  # Initial speed
angles = {angles.tolist()}  # Launch angles

fig = plt.figure(figsize=(10, 5))
for theta in angles:
    x, y, t_flight = projectile_motion(v0, theta)
    plt.plot(x, y, label=f"{{theta}}°")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.title(f"Projectile Motion (v0 = {{v0}} m/s)")
plt.legend()
plt.grid()
plt.show()'''
        design_id = design_dict[design_choice]
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text, include_name, design_id)
        
        # Display the generated PDF (with standard controls)
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_iframe = f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe></div>'
        st.markdown(pdf_iframe, unsafe_allow_html=True)
        
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="projectile_report.pdf")

elif experiment == "Rainfall Probability Estimation":
    st.header("Rainfall Probability Analysis")
    if st.button("Generate and Preview Report"):
        mu = np.random.randint(30, 71)
        sigma = np.random.randint(5, 21)
        pdf_func = lambda x: norm.pdf(x, mu, sigma)
        prob, _ = spi.quad(pdf_func, 30, 70)

        fig = plt.figure(figsize=(8, 5))
        x = np.linspace(0, 100, 1000)
        y = pdf_func(x)
        plt.plot(x, y, label="Rainfall Distribution")
        plt.fill_between(x, y, where=(x >= 30) & (x <= 70), color="green", alpha=0.5)
        plt.xlabel("Rainfall (mm)")
        plt.ylabel("Probability Density")
        plt.title(f"Probability (30mm-70mm): {prob:.2f} (mu={mu}, sigma={sigma})")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        results_text = f"Normal Distribution Parameters:\nMean (mu): {mu}\nStd Dev (sigma): {sigma}\nProbability (30mm-70mm): {prob:.4f}"
        code = f'''import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = {mu}  # Mean rainfall
sigma = {sigma}  # Std Dev
pdf = lambda x: norm.pdf(x, mu, sigma)
prob, _ = spi.quad(pdf, 30, 70)

x = np.linspace(0, 100, 1000)
y = pdf(x)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Rainfall Distribution")
plt.fill_between(x, y, where=(x>=30)&(x<=70), color="green", alpha=0.5)
plt.xlabel("Rainfall (mm)")
plt.ylabel("Probability Density")
plt.title(f"Probability: {{prob:.2f}}")
plt.legend()
plt.grid()
plt.show()'''
        design_id = design_dict[design_choice]
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text, include_name, design_id)
        
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_iframe = f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe></div>'
        st.markdown(pdf_iframe, unsafe_allow_html=True)
        
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="rainfall_report.pdf")

elif experiment == "3D Hit Probability Distribution":
    st.header("3D Hit Probability Analysis")
    if st.button("Generate and Preview Report"):
        mean = np.random.randint(0, 31, size=3)
        A = np.random.randint(1, 5, (3, 3))
        cov = np.dot(A, A.T)
        
        data = np.random.multivariate_normal(mean, cov, 1000)
        x, y, z = data.T

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, alpha=0.5)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Hit Probability Distribution")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        results_text = "Mean Vector:\n" + f"{mean.tolist()}\n\nCovariance Matrix:\n"
        for row in cov:
            results_text += f"{row.tolist()}\n"
        
        code = f'''import numpy as np
import matplotlib.pyplot as plt

mean = {mean.tolist()}  # Mean vector for 3D distribution
cov = {[row.tolist() for row in cov]}   # Covariance matrix

data = np.random.multivariate_normal(mean, cov, 1000)
x, y, z = data.T
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, alpha=0.5)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Hit Probability Distribution")
plt.show()'''
        design_id = design_dict[design_choice]
        pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text, include_name, design_id)
        
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_iframe = f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe></div>'
        st.markdown(pdf_iframe, unsafe_allow_html=True)
        
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="3d_probability_report.pdf")
ph_img, code, results, include_name):
    """
    Creates a styled PDF report with user-controlled inclusion of Name, USN, and Section.
    """
    class PDF(FPDF):
        def header(self):
            # Header styling
            self.set_fill_color(0, 128, 128)
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 20)
            self.cell(0, 15, "MATHEMATICS INNOVATIVE EXPERIMENT", ln=1, align='C', fill=True)
            self.ln(3)
            self.set_line_width(1.5)
            self.set_draw_color(0, 0, 0)
            self.rect(5, 5, self.w - 10, self.h - 10)
            self.set_text_color(0, 0, 0)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # ----------------------- QUESTION Section -----------------------
    pdf.set_fill_color(200, 230, 201)  # light green
    pdf.set_text_color(34, 139, 34)    # forest green
    pdf.set_font('Times', 'B', 16)
    pdf.cell(0, 12, "QUESTION:", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font('Times', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, experiment)
    pdf.ln(4)
    
    # ----------------------- Student Information Section -----------------------
    if include_name:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(230, 230, 250)  # lavender fill
        pdf.cell(0, 10, f"NAME: {name}", ln=True, align="C", fill=True)
        pdf.cell(0, 10, f"USN: {usn}", ln=True, align="C", fill=True)
        pdf.cell(0, 10, f"SECTION: {section}", ln=True, align="C", fill=True)
        pdf.ln(4)
    
    # ----------------------- EXPERIMENT CODE Section -----------------------
    pdf.set_fill_color(255, 228, 196)  # bisque fill
    pdf.set_font('Courier', 'B', 14)
    pdf.cell(0, 10, "EXPERIMENT CODE:", ln=True, fill=True)
    pdf.set_font('Courier', '', 10)
    pdf.multi_cell(0, 5, code)
    pdf.ln(4)
    
    # ----------------------- OUTPUT WITH GRAPH Section -----------------------
    pdf.set_fill_color(255, 240, 245)  # lavender blush fill
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "OUTPUT WITH GRAPH:", ln=True, fill=True)
    pdf.ln(2)
    if graph_img:
        img_path = "graph.png"
        with open(img_path, "wb") as f:
            f.write(graph_img.getbuffer())
        pdf.image(img_path, x=20, w=pdf.w - 40)
        pdf.ln(5)
        os.remove(img_path)
    
    # ----------------------- RESULTS Section -----------------------
    pdf.set_fill_color(255, 250, 205)  # lemon chiffon fill
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "RESULTS:", ln=True, fill=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, results)
    
    pdf_file = "report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# ----------------------- Generate PDF for Selected Experiment -----------------------
if st.button("Generate Report"):
    # Random input values
    v0 = np.random.randint(10, 31)
    angles = np.random.randint(20, 71, size=3)
    g = 9.81

    def projectile_motion(v0, theta, g=9.81):
        theta_rad = np.radians(theta)
        t_flight = 2 * v0 * np.sin(theta_rad) / g
        t = np.linspace(0, t_flight, 100)
        x = v0 * np.cos(theta_rad) * t
        y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
        return x, y, t_flight

    fig = plt.figure(figsize=(10, 5))
    results_lines = []
    for theta in angles:
        x, y, t_flight = projectile_motion(v0, theta)
        plt.plot(x, y, label=f"{theta}°")
        max_height = (v0**2 * np.sin(np.radians(theta))**2) / (2 * g)
        range_val = (v0**2 * np.sin(2 * np.radians(theta))) / g
        results_lines.append(f"Angle {theta}°: Time: {t_flight:.2f}s | Max Height: {max_height:.2f}m | Range: {range_val:.2f}m")
    
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Vertical Distance (m)")
    plt.title(f"Projectile Motion (v0 = {v0} m/s)")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    results_text = f"Initial Speed (v0): {v0} m/s\n" + "\n".join(results_lines)
    code = f'''import numpy as np
import matplotlib.pyplot as plt
def projectile_motion(v0, theta, g=9.81):
    theta_rad = np.radians(theta)
    t_flight = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_flight, 100)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return x, y, t_flight
v0 = {v0}  
angles = {angles.tolist()}
plt.figure(figsize=(10, 5))
for theta in angles:
    x, y, t_flight = projectile_motion(v0, theta)
    plt.plot(x, y, label=f"{{theta}}°")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.legend()
plt.grid()
plt.show()'''

    pdf_path = create_pdf(name, usn, section, experiment, buf, code, results_text, include_name)
    with open(pdf_path, "rb") as f:
        st.download_button("Download Report", f, file_name="experiment_report.pdf")
