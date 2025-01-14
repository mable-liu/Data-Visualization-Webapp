import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Page Title
st.title("Santa's Curve Fitting Workshop 🎅")

# Initialize session state for snow effect
if "show_snow" not in st.session_state:
    st.session_state.show_snow = False

# Tabs for Data Input
tab1, tab2 = st.tabs(["Enter Data by Hand", "Upload CSV File"])

x_data = None
y_data = None

# Manual Entry Tab
with tab1:
    st.header("Manual Data Entry")
    x_values = st.text_area("Enter X values (comma-separated)", "1, 2, 3, 4, 5", key="x_values")
    y_values = st.text_area("Enter Y values (comma-separated)", "1, 4, 9, 16, 25", key="y_values")

    # Trigger snow if X or Y values change
    if st.session_state.x_values or st.session_state.y_values:
        try:
            x_data = np.array([float(x.strip()) for x in st.session_state.x_values.split(",")])
            y_data = np.array([float(y.strip()) for y in st.session_state.y_values.split(",")])
            df = pd.DataFrame({'X': x_data, 'Y': y_data})

            # Snow effect when values are entered or changed
            if not st.session_state.show_snow:
                st.snow()
                st.session_state.show_snow = True
        except ValueError:
            st.error("Please enter valid numeric values for X and Y.")

# CSV Upload Tab
with tab2:
    st.header("Upload CSV File")
    st.success("Please ensure the CSV file has exactly 2 columns (X, Y) and no header.")
    uploaded_file = st.file_uploader("", type=["csv"])
    if uploaded_file:
        try:
            # Load CSV file and check its structure
            data = np.loadtxt(uploaded_file, delimiter=",")
            if data.shape[1] != 2:
                st.error("The uploaded file must have exactly 2 columns (X and Y).")
            else:
                x_data = data[:, 0]
                y_data = data[:, 1]
                df = pd.DataFrame({'X': x_data, 'Y': y_data})

                # Snow effect when a file is uploaded
                st.snow()
        except Exception as e:
            st.error(f"Error processing the file: {e}")


# Sidebar: Input Data Table and Summary
if x_data is not None and y_data is not None:
    with st.sidebar:
        st.header("Graph Statistics 🎄")  # Updated title
        st.markdown("---")  # Underline effect
        st.subheader("Input Data")
        st.table(df)
        st.subheader("Data Summary")  # Updated title
        st.write(df.describe())

# Curve Fitting Options and Plot
if x_data is not None and y_data is not None:
    st.header("Curve Fitting Options :snowflake:")

    # Choose Fit Type
    fit_type = st.selectbox("Choose the type of curve to fit", ["Polynomial", "Exponential", "Logarithmic"])

    # Initialize Plot
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, label="Data Points", color="red")

    # Initialize equation string and y_fit
    equation = ""
    y_fit = None

    if fit_type == "Polynomial":
        degree = st.slider("Degree of Polynomial", 1, 10, 2)
        try:
            coeffs = np.polyfit(x_data, y_data, degree)
            y_fit = np.polyval(coeffs, x_data)
            ax.plot(x_data, y_fit, label=f"Polynomial Fit (degree {degree})", color="green")

            # Construct polynomial equation string
            terms = [f"{coeffs[i]:.3f}x^{degree - i}" for i in range(degree)]
            terms.append(f"{coeffs[-1]:.3f}")
            equation = " + ".join(terms).replace("x^1", "x")
            equation = equation.replace("x^0", "").replace("+ -", "- ")

            # Display Fitted Coefficients in Horizontal Chart
            st.subheader("Fitted Coefficients")
            coeff_df = pd.DataFrame([coeffs], columns=[f"Coeff {i+1}" for i in range(len(coeffs))])
            st.table(coeff_df)
        except Exception as e:
            st.error(f"Could not fit polynomial: {e}")

    elif fit_type == "Exponential":
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c
        try:
            popt, _ = curve_fit(exponential, x_data, y_data)
            y_fit = exponential(x_data, *popt)
            ax.plot(x_data, y_fit, label="Exponential Fit", color="green")
            equation = f"{popt[0]:.3f}e^({popt[1]:.3f}x) + {popt[2]:.3f}"

            # Display Fitted Parameters in Horizontal Chart
            st.subheader("Fitted Parameters")
            param_df = pd.DataFrame([popt], columns=["a", "b", "c"])
            st.table(param_df)
        except Exception as e:
            st.error(f"Could not fit exponential curve: {e}")

    elif fit_type == "Logarithmic":
        def logarithmic(x, a, b):
            return a * np.log(x) + b
        try:
            if np.any(x_data <= 0):
                st.error("Logarithmic fit requires all X values to be positive.")
            else:
                popt, _ = curve_fit(logarithmic, x_data, y_data)
                y_fit = logarithmic(x_data, *popt)
                ax.plot(x_data, y_fit, label="Logarithmic Fit", color="green")
                equation = f"{popt[0]:.3f}log(x) + {popt[1]:.3f}"

                # Display Fitted Parameters in Horizontal Chart
                st.subheader("Fitted Parameters")
                param_df = pd.DataFrame([popt], columns=["a", "b"])
                st.table(param_df)
        except Exception as e:
            st.error(f"Could not fit logarithmic curve: {e}")

    # Calculate and Display Fit Quality
    if y_fit is not None:
        residuals = y_data - y_fit
        mse = np.mean(residuals**2)  # Mean Squared Error
        mae = np.mean(np.abs(residuals))  # Mean Absolute Error

        # Display metrics in Sidebar
        with st.sidebar:
            st.success(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.success(f"Mean Squared Error (MSE): {mse:.4f}")

        # Display equation above the graph
        st.subheader("Equation of Fitted Curve")
        st.latex(f"y = {equation}")
        
    ax.grid(True)  # Add grid to the fitted curve graph

    # Finalize and Show the Plot
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)

   # Option to Fit Statistical Distribution
    fit_distribution = st.toggle("Display Histogram", value=False)
    if fit_distribution:
        st.header("Distribution Histogram")
        try:
            # Count frequency of each unique x value
            unique_x, counts = np.unique(x_data, return_counts=True)
    
            # Fit a Gaussian to the raw X data
            mu, std = norm.fit(x_data)
    
            # Plot the histogram using primary y-axis
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(unique_x, weights=counts, bins=len(unique_x), density=False, alpha=0.6, color="green", edgecolor="black")
            ax_hist.set_xlabel("X Value")
            ax_hist.set_ylabel("Frequency")
            ax_hist.set_title("Histogram")
    
            # Twin the axes to plot the Gaussian curve
            ax_gaussian = ax_hist.twinx()
            xmin, xmax = ax_hist.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax_gaussian.plot(x, p, 'k', linewidth=2, label="Gaussian Fit")
            ax_gaussian.set_ylabel("Density")
            ax_gaussian.legend(loc='upper right')

    
            st.pyplot(fig_hist)
    
            # Display fitted parameters
            st.subheader("Fitted Parameters for Histogram")
            st.write(f"Mean (μ): {mu:.3f}")
            st.write(f"Standard Deviation (σ): {std:.3f}")
        except Exception as e:
            st.error(f"Could not fit Gaussian distribution: {e}")

else:
    st.warning("Please provide valid data to fit curves.")
