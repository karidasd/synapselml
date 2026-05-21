import streamlit as st
import time
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

# Set Streamlit page settings
st.set_page_config(
    page_title="SynapseLML // Real-time Latent Synchronization Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphic dark UI
st.markdown("""
<style>
    /* Dark background for the whole app */
    .stApp {
        background-color: #0d0f18 !important;
        color: #e2e8f0 !important;
    }
    
    /* Side panel background */
    section[data-testid="stSidebar"] {
        background-color: #08090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Metric card design */
    .metric-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-top: 5px;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Pulse indicator animation */
    .pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.5s infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "synapselml_telemetry.pkl")

@st.cache_resource
def get_stable_pca() -> PCA:
    """
    Fits a stable, globally oriented PCA using simulated target states.
    This guarantees that the 3D coordinate system remains fixed and doesn't
    rotate randomly across updates, allowing smooth docking visual movement.
    """
    torch.manual_seed(42)
    true_mapping = torch.empty(256, 512)
    nn.init.orthogonal_(true_mapping)
    true_bias = torch.randn(512) * 0.02
    
    samples = []
    # Build a representative sample across 20 epochs of variance
    for epoch in range(1, 21):
        temporal_bias = torch.sin(torch.tensor(epoch * 0.4)).item()
        analyst_states = torch.randn(4, 16, 256) * 0.5 + temporal_bias
        executor_expected = torch.matmul(analyst_states, true_mapping) + true_bias
        samples.append(executor_expected.numpy())
        
    flat_samples = np.vstack(samples).reshape(-1, 512)
    pca = PCA(n_components=3)
    pca.fit(flat_samples)
    return pca

# Sidebar Configuration
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #38bdf8;'>🌌 SynapseLML Control</h2>", 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Protocol Specifications")
st.sidebar.markdown("**Vector Transport Protocol (VTP)**")
st.sidebar.markdown("- Client: Analyst Agent ($d_1 = 256$)")
st.sidebar.markdown("- Server: Executor Agent ($d_2 = 512$)")
st.sidebar.markdown("- Compression: FP8 Symmetric Quantization")
st.sidebar.markdown("- Optimizer: Online Entropy-Based Runtime (EBR)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Config")
refresh_rate = st.sidebar.slider("Polling Frequency (seconds)", 0.1, 2.0, 0.3)
auto_rotate = st.sidebar.checkbox("Auto-Rotate 3D Space", value=True)

# Main Dashboard Title
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 5px; color: #f8fafc;'>"
    "🌌 SynapseLML Silicon Telepathy Dashboard</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 25px;'>"
    "Decentralized Multi-Process Latent Space Synchronization (LSS) Visualization</p>",
    unsafe_allow_html=True
)

# Placeholder container for the dynamic UI
placeholder = st.empty()

# Fetch the global stable PCA projection
pca_projector = get_stable_pca()

# Initialize last modification time tracking
last_mtime = 0.0

while True:
    if not os.path.exists(TELEMETRY_PATH):
        with placeholder.container():
            st.warning("⚠️ Waiting for telemetry stream from the socket server...")
            st.info(
                "To start the simulation:\n\n"
                "1. Run the Executor Server:\n"
                "   `python synapselml/network/server.py`\n\n"
                "2. Run the Analyst Client:\n"
                "   `python synapselml/network/client.py`"
            )
            time.sleep(1.0)
            continue
            
    # Read telemetry data
    try:
        current_mtime = os.path.getmtime(TELEMETRY_PATH)
        if current_mtime == last_mtime:
            time.sleep(refresh_rate)
            continue
            
        with open(TELEMETRY_PATH, "rb") as f:
            data = pickle.load(f)
            
        last_mtime = current_mtime
    except Exception:
        # File might be locked during writing, retry shortly
        time.sleep(0.05)
        continue

    # Extract states and metrics
    epoch = data["epoch"]
    source_states = data["source_states"]  # Shape: [B, L, 256]
    projected_states = data["projected_states"]  # Shape: [B, L, 512]
    target_states = data["target_states"]  # Shape: [B, L, 512]
    metrics = data["metrics"]
    history = data["history"]
    
    # Process coordinates for Plotly 3D plot
    B, L, d_source = source_states.shape
    _, _, d_target = target_states.shape
    
    # Reshape matrices to 2D for PCA projection
    projected_flat = projected_states.reshape(-1, d_target)
    target_flat = target_states.reshape(-1, d_target)
    
    # Project down to stable 3D space
    projected_3d = pca_projector.transform(projected_flat)
    target_3d = pca_projector.transform(target_flat)
    
    # Build connections/alignment vector lines
    x_lines = []
    y_lines = []
    z_lines = []
    for i in range(len(target_3d)):
        x_lines.extend([target_3d[i, 0], projected_3d[i, 0], None])
        y_lines.extend([target_3d[i, 1], projected_3d[i, 1], None])
        z_lines.extend([target_3d[i, 2], projected_3d[i, 2], None])

    # Dynamic UI assembly inside the placeholder
    with placeholder.container():
        # Top Row Metric Panel
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{metrics['loss']:.6f}</div>"
                f"<div class='metric-label'>Total Alignment Loss</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{metrics['mse']:.6f}</div>"
                f"<div class='metric-label'>Mean Squared Error</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{metrics['cosine_distance']:.6f}</div>"
                f"<div class='metric-label'>Cosine Distance</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{metrics['orthogonal_penalty']:.6f}</div>"
                f"<div class='metric-label'>Orthogonal Penalty</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Canvas & Details Split
        main_col, side_col = st.columns([7, 3])
        
        with main_col:
            st.markdown(
                f"### <span class='pulse'></span> Real-time LSS Point Clouds (Epoch {epoch})", 
                unsafe_allow_html=True
            )
            
            # Setup Plotly Traces
            target_trace = go.Scatter3d(
                x=target_3d[:, 0],
                y=target_3d[:, 1],
                z=target_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color='#00ffff',
                    opacity=0.8,
                    symbol='circle',
                    line=dict(color='rgba(0, 255, 255, 0.4)', width=1)
                ),
                name='Target Agent Space (H_B)'
            )

            projected_trace = go.Scatter3d(
                x=projected_3d[:, 0],
                y=projected_3d[:, 1],
                z=projected_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color='#ff007f',
                    opacity=0.8,
                    symbol='circle',
                    line=dict(color='rgba(255, 0, 127, 0.4)', width=1)
                ),
                name='Projected Source (H_B_hat)'
            )

            vector_trace = go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode='lines',
                line=dict(
                    color='rgba(200, 200, 255, 0.25)',
                    width=1.5
                ),
                name='Alignment Discrepancy'
            )

            # Determine plot boundary limits
            max_val = max(np.max(np.abs(target_3d)), np.max(np.abs(projected_3d))) * 1.15 if len(target_3d) > 0 else 2.0
            
            camera_dict = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.35, y=1.35, z=0.9)
            )
            
            if auto_rotate:
                # Dynamically rotate camera slightly using epoch counter
                angle = epoch * 0.08
                camera_dict['eye'] = dict(
                    x=1.5 * np.cos(angle),
                    y=1.5 * np.sin(angle),
                    z=0.9
                )

            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        range=[-max_val, max_val], 
                        backgroundcolor="rgb(10, 12, 20)",
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        showbackground=True,
                        title="PCA Axis 1"
                    ),
                    yaxis=dict(
                        range=[-max_val, max_val], 
                        backgroundcolor="rgb(10, 12, 20)",
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        showbackground=True,
                        title="PCA Axis 2"
                    ),
                    zaxis=dict(
                        range=[-max_val, max_val], 
                        backgroundcolor="rgb(10, 12, 20)",
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        showbackground=True,
                        title="PCA Axis 3"
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1),
                    camera=camera_dict
                ),
                margin=dict(r=0, l=0, b=0, t=0),
                paper_bgcolor='rgba(13, 15, 24, 1)',
                plot_bgcolor='rgba(13, 15, 24, 1)',
                template='plotly_dark',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(10, 12, 20, 0.8)",
                    bordercolor="rgba(255, 255, 255, 0.1)",
                    borderwidth=1
                ),
                height=650
            )

            fig = go.Figure(data=[vector_trace, target_trace, projected_trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_lss_epoch_{epoch}")

        with side_col:
            st.markdown("### 📈 Alignment History")
            
            # Convert history to pandas dataframe for graphing
            df_history = pd.DataFrame(history)
            
            if not df_history.empty:
                # Loss convergence chart
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=df_history["epoch"],
                    y=df_history["loss"],
                    mode='lines+markers',
                    name='Total Loss',
                    line=dict(color='#38bdf8', width=2),
                    marker=dict(size=4)
                ))
                fig_loss.update_layout(
                    margin=dict(r=10, l=10, b=20, t=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    template='plotly_dark',
                    height=280,
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Epoch"),
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Loss"),
                    showlegend=False
                )
                st.plotly_chart(fig_loss, use_container_width=True, key="plotly_loss_history")
                
                # Orthogonal Penalty track
                fig_orth = go.Figure()
                fig_orth.add_trace(go.Scatter(
                    x=df_history["epoch"],
                    y=df_history["orthogonal_penalty"],
                    mode='lines+markers',
                    name='Orth Penalty',
                    line=dict(color='#a855f7', width=2),
                    marker=dict(size=4)
                ))
                fig_orth.update_layout(
                    margin=dict(r=10, l=10, b=20, t=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    template='plotly_dark',
                    height=240,
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Epoch"),
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Orth Penalty"),
                    showlegend=False
                )
                st.plotly_chart(fig_orth, use_container_width=True, key="plotly_orth_history")

            # Backpropagation updates tracker
            bp_count = df_history["micro_backprop_triggered"].sum() if not df_history.empty else 0
            st.markdown(
                f"<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); "
                f"border-radius: 8px; padding: 12px; text-align: center; margin-top: 10px;'>"
                f"<span style='font-size: 14px; font-weight: bold; color: #10b981;'>"
                f"⚡ Active Inline Corrections (Micro-BP): {int(bp_count)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    time.sleep(refresh_rate)
