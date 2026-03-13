import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

st.set_page_config(page_title="McCabe-Thiele Auto-Stepper", layout="wide")

def main():
    st.title("Distillation Column McCabe-Thiele Auto-Stepper")
    
    # --- SIDEBAR: DATA INGESTION ---
    st.sidebar.header("1. VLE Data Input")
    data_option = st.sidebar.selectbox(
        "Choose VLE Data Source:",
        ["Constant Relative Volatility", "Antoine Equations", "Copy/Paste Arrays", "CSV Upload"]
    )

    x_data, y_data = None, None

    if data_option == "Constant Relative Volatility":
        alpha = st.sidebar.slider("Relative Volatility (alpha)", 1.1, 5.0, 2.5, 0.1)
        x_data = np.linspace(0, 1, 100)
        y_data = (alpha * x_data) / (1 + (alpha - 1) * x_data)

    elif data_option == "Antoine Equations":
        st.sidebar.markdown("Using log10(P) = A - (B / (T + C))")
        P_sys = st.sidebar.number_input("System Pressure", value=760.0)
        st.sidebar.subheader("More Volatile Component")
        A1 = st.sidebar.number_input("A1", value=7.96)
        B1 = st.sidebar.number_input("B1", value=1668.21)
        C1 = st.sidebar.number_input("C1", value=228.0)
        st.sidebar.subheader("Less Volatile Component")
        A2 = st.sidebar.number_input("A2", value=8.15)
        B2 = st.sidebar.number_input("B2", value=1810.94)
        C2 = st.sidebar.number_input("C2", value=227.1)
        
        try:
            # Calculate saturation temperatures
            T1_sat = B1 / (A1 - np.log10(P_sys)) - C1
            T2_sat = B2 / (A2 - np.log10(P_sys)) - C2
            T_array = np.linspace(min(T1_sat, T2_sat), max(T1_sat, T2_sat), 100)
            
            P1_sat = 10 ** (A1 - (B1 / (T_array + C1)))
            P2_sat = 10 ** (A2 - (B2 / (T_array + C2)))
            
            x_data = (P_sys - P2_sat) / (P1_sat - P2_sat)
            y_data = (x_data * P1_sat) / P_sys
            
            # Sort data for interpolation
            idx = np.argsort(x_data)
            x_data, y_data = x_data[idx], y_data[idx]
        except Exception as e:
            st.sidebar.error("Error calculating from Antoine coefficients. Check inputs.")

    elif data_option == "Copy/Paste Arrays":
        x_str = st.sidebar.text_area("x-array (comma separated)", "0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0")
        y_str = st.sidebar.text_area("y-array (comma separated)", "0.0, 0.2, 0.5, 0.7, 0.85, 0.95, 1.0")
        try:
            x_data = np.array([float(i.strip()) for i in x_str.split(',')])
            y_data = np.array([float(i.strip()) for i in y_str.split(',')])
        except:
            st.sidebar.error("Invalid array format. Please use comma-separated numbers.")

    elif data_option == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns must be named 'x' and 'y')", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'x' in df.columns and 'y' in df.columns:
                x_data = df['x'].values
                y_data = df['y'].values
            else:
                st.sidebar.error("CSV must contain 'x' and 'y' columns.")

    # Stop execution if valid data isn't ready
    if x_data is None or y_data is None:
        st.warning("Please provide valid VLE data to proceed.")
        return

    # --- SIDEBAR: OPERATING CONDITIONS ---
    st.sidebar.header("2. Operating Conditions")
    
    # Base parameters required for all methods
    xD = st.sidebar.slider("Distillate Purity (xD)", 0.50, 0.99, 0.95, 0.01)
    xB = st.sidebar.slider("Bottoms Purity (xB)", 0.01, 0.50, 0.05, 0.01)
    zF = st.sidebar.slider("Feed Composition (zF)", xB + 0.01, xD - 0.01, 0.50, 0.01)
    q = st.sidebar.slider("Feed Quality (q)", -0.5, 1.5, 1.0, 0.1)

    # Let the user choose what defines their column's internal flows
    spec_method = st.sidebar.selectbox(
        "Column Specification Method:",
        ["Known Reflux Ratio (R)", "Multiple of Minimum Reflux", "Known Boilup Ratio (Vb)"]
    )

    # Conditionally show sliders based on the chosen method
    if spec_method == "Known Reflux Ratio (R)":
        R = st.sidebar.slider("Reflux Ratio (R)", 0.1, 10.0, 2.0, 0.1)
        # Proceed with standard rectifying line calculation...

    elif spec_method == "Multiple of Minimum Reflux":
        r_mult = st.sidebar.slider("R / R_min Multiplier", 1.01, 3.0, 1.2, 0.05)
        
        # 1. Define a function for the difference between VLE and q-line
        def q_line_diff(x):
            y_vle = f_y_vle(x)
            if abs(q - 1.0) < 1e-5: # Saturated liquid
                return x - zF
            else:
                m_q = q / (q - 1)
                b_q = -zF / (q - 1)
                y_q = m_q * x + b_q
                return y_vle - y_q

        # 2. Find the pinch point (intersection of q-line and VLE)
        try:
            # We search for the intersection between xB and xD
            res = root_scalar(q_line_diff, bracket=[xB, xD], method='brentq')
            x_pinch = res.root
            y_pinch = float(f_y_vle(x_pinch))
            
            # 3. Calculate minimum slope and R_min
            m_min = (xD - y_pinch) / (xD - x_pinch)
            R_min = m_min / (1 - m_min)
            
            # 4. Calculate actual R
            R = r_mult * R_min
            st.sidebar.success(f"Calculated R_min: {R_min:.2f}  |  Actual R: {R:.2f}")
            
        except ValueError:
            st.sidebar.error("Could not find a valid pinch point. Check zF and q values.")
            R = 2.0 # Fallback value so the app doesn't crash completely

    elif spec_method == "Known Boilup Ratio (Vb)":
        Vb = st.sidebar.slider("Boilup Ratio (Vb)", 0.1, 10.0, 2.0, 0.1)
        # Add calculation logic here:
        # 1. Calculate stripping line slope: m_S = (Vb + 1) / Vb
        # 2. Find intersection of stripping line and q-line
        # 3. Calculate rectifying line from xD to that intersection

    # --- CALCULATIONS ---
    # Interpolation functions for VLE curve
    try:
        f_y_vle = interp1d(x_data, y_data, kind='cubic', fill_value="extrapolate")
        f_x_vle = interp1d(y_data, x_data, kind='cubic', fill_value="extrapolate")
    except ValueError:
        st.error("VLE data is not strictly increasing. Please check your inputs.")
        return

    # Rectifying line: y = (R/(R+1))x + xD/(R+1)
    m_R = R / (R + 1)
    b_R = xD / (R + 1)

    # q-line and Intersection
    if abs(q - 1.0) < 1e-5: # Saturated Liquid
        x_int = zF
        y_int = m_R * x_int + b_R
    else:
        m_q = q / (q - 1)
        b_q = -zF / (q - 1)
        x_int = (b_q - b_R) / (m_R - m_q)
        y_int = m_R * x_int + b_R

    # Stripping line (passes through (xB, xB) and (x_int, y_int))
    m_S = (y_int - xB) / (x_int - xB)
    b_S = xB - m_S * xB

    def op_line_y(x_val):
        if x_val > x_int:
            return m_R * x_val + b_R
        else:
            return m_S * x_val + b_S

    # --- STEPPING ALGORITHM ---
    stages_x = [xD]
    stages_y = [xD]
    current_x = xD
    current_y = xD
    stage_count = 0

    while current_x > xB and stage_count < 50: # Cap at 50 stages to prevent infinite loops
        # Step horizontal to VLE curve
        next_x = float(f_x_vle(current_y))
        stages_x.append(next_x)
        stages_y.append(current_y)
        
        if next_x < xB:
            # We've crossed the bottoms target, finish the last step visually
            stages_x.append(next_x)
            stages_y.append(next_x)
            stage_count += 1
            break
            
        # Step vertical to operating line
        next_y = op_line_y(next_x)
        stages_x.append(next_x)
        stages_y.append(next_y)
        
        current_x = next_x
        current_y = next_y
        stage_count += 1

    # --- PLOTTING ---
    fig = go.Figure()

    # VLE Curve and y=x line
    x_plot = np.linspace(0, 1, 100)
    fig.add_trace(go.Scatter(x=x_plot, y=f_y_vle(x_plot), mode='lines', name='Equilibrium Curve', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='y = x', line=dict(color='grey', dash='dash')))

    # Operating Lines
    fig.add_trace(go.Scatter(x=[xD, x_int], y=[xD, y_int], mode='lines', name='Rectifying Line', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[x_int, xB], y=[y_int, xB], mode='lines', name='Stripping Line', line=dict(color='red')))
    
    # q-line
    # Determine where q-line hits y=x
    fig.add_trace(go.Scatter(x=[zF, x_int], y=[zF, y_int], mode='lines', name='q-line', line=dict(color='purple', dash='dot')))

    # Stages
    fig.add_trace(go.Scatter(x=stages_x, y=stages_y, mode='lines', name='Stages', line=dict(color='black')))

    fig.update_layout(
        title="McCabe-Thiele Diagram",
        xaxis_title="Liquid Mole Fraction (x)",
        yaxis_title="Vapor Mole Fraction (y)",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        height=700,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"Theoretical Stages Required: {stage_count}")

if __name__ == "__main__":
    main()
