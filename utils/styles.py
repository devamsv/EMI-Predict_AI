import streamlit as st

def apply_styling():
    """
    Applies custom CSS for a clean white theme, maximum readability,
    professional card styling, and custom animations.
    """
    st.markdown(
        """
        <style>
        /* Global Reset & Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1f2937; /* Dark Gray text */
            background-color: #ffffff;
        }

        /* Main App Background - Pure White */
        .stApp {
            background-color: #ffffff;
            color: #1f2937;
        }

        /* Headings - Dark Blue/Slate */
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a;
            font-weight: 700;
        }
        
        /* Card Styling (Clean White) */
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb; /* Light Gray border */
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); /* Soft shadow */
            transition: box-shadow 0.2s ease;
        }
        
        .card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        }

        /* Metric Cards */
        .metric-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid #e5e7eb;
            border-top: 4px solid #2563eb; /* Primary Blue Accent */
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Button Styling - Primary Blue */
        .stButton > button {
            background-color: #2563eb;
            color: #ffffff;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 10px 24px;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }

        /* Chatbot Message Styling */
        .chat-user {
            background-color: #eff6ff; /* Light Blue BG */
            padding: 12px 16px;
            border-radius: 12px 12px 0 12px;
            margin-bottom: 12px;
            text-align: right;
            color: #1e3a8a;
            border: 1px solid #dbeafe;
        }
        
        .chat-ai {
            background-color: #f8fafc; /* Light Gray BG */
            padding: 12px 16px;
            border-radius: 12px 12px 12px 0;
            margin-bottom: 12px;
            text-align: left;
            color: #334155;
            border: 1px solid #e2e8f0;
        }

        /* Cut-Paper Animation Effect */
        @keyframes slideUpFade {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .paper-card {
            background-color: #ffffff;
            border-radius: 2px; /* Sharper corners for paper look */
            border: 1px solid #e2e8f0;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            animation: slideUpFade 0.6s ease-out forwards;
            position: relative;
        }
        
        /* Paper Fold effect visual */
        .paper-card::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            border-width: 0 40px 40px 0;
            border-style: solid;
            border-color: #f1f5f9 #ffffff;
            box-shadow: -5px 5px 5px rgba(0,0,0,0.05);
            display: none; /* Optional: Enable if specific fold design desired */
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

def render_hero_section(title, subtitle):
    """
    Renders a clean, professional hero section.
    """
    st.markdown(
        f"""
        <div class="card" style="text-align: center; padding: 48px 20px; background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%); border: 1px solid #dbeafe;">
            <h1 style="font-size: 2.5rem; color: #1e40af; margin-bottom: 12px; font-weight: 800; letter-spacing: -0.025em;">
                {title}
            </h1>
            <p style="font-size: 1.125rem; color: #475569; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
