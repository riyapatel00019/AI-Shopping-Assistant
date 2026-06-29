"""
============================================================
ShopWise AI
Modern AI Shopping Assistant
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import time
import requests
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="ShopWise AI",

    page_icon="🛍️",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ============================================================
# API
# ============================================================

API_URL = "http://127.0.0.1:8000/chat"

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "shopping_context" not in st.session_state:

    st.session_state.shopping_context = {}

if "conversation_complete" not in st.session_state:

    st.session_state.conversation_complete = True

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""

<style>

/* ---------------------------------------------------- */
/* Hide Streamlit */
/* ---------------------------------------------------- */

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* ---------------------------------------------------- */
/* Main */
/* ---------------------------------------------------- */

html,
body,
[data-testid="stAppViewContainer"]{

background:#0b1220;
color:white;

}

.block-container{

padding-top:20px;
padding-left:25px;
padding-right:25px;
max-width:1500px;

}

/* ---------------------------------------------------- */
/* Sidebar */
/* ---------------------------------------------------- */

[data-testid="stSidebar"]{

background:#111827;

}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p{

color:white;

}

/* ---------------------------------------------------- */
/* Header */
/* ---------------------------------------------------- */

.main-title{

font-size:42px;

font-weight:800;

color:white;

margin-bottom:0px;

}

.sub-title{

font-size:18px;

color:#94A3B8;

margin-top:-5px;

margin-bottom:20px;

}

/* ---------------------------------------------------- */
/* Chat */
/* ---------------------------------------------------- */

[data-testid="stChatMessage"]{

background:#161F2F;

border-radius:18px;

padding:15px;

margin-bottom:15px;

border:1px solid #253047;

}

/* ---------------------------------------------------- */
/* Product Card */
/* ---------------------------------------------------- */

.product-card{

background:#161F2F;

border-radius:18px;

padding:18px;

border:1px solid #2D3B55;

margin-bottom:20px;

transition:0.3s;

}

.product-card:hover{

border:1px solid #4F8EF7;

box-shadow:0 0 15px rgba(79,142,247,.35);

}

/* ---------------------------------------------------- */
/* Product Title */
/* ---------------------------------------------------- */

.product-title{

font-size:22px;

font-weight:bold;

color:white;

margin-top:10px;

}

/* ---------------------------------------------------- */
/* Brand */
/* ---------------------------------------------------- */

.brand{

display:inline-block;

background:#2563EB;

padding:5px 12px;

border-radius:10px;

font-size:13px;

color:white;

margin-top:8px;

}

/* ---------------------------------------------------- */
/* Price */
/* ---------------------------------------------------- */

.price{

display:inline-block;

background:#16A34A;

padding:7px 12px;

border-radius:10px;

font-weight:bold;

font-size:18px;

color:white;

margin-top:12px;

}

/* ---------------------------------------------------- */
/* Rating */
/* ---------------------------------------------------- */

.rating{

font-size:18px;

color:#FFD700;

margin-top:10px;

}

/* ---------------------------------------------------- */
/* Why Recommended */
/* ---------------------------------------------------- */

.reason{

background:#0F172A;

padding:14px;

border-left:4px solid #22C55E;

border-radius:10px;

margin-top:15px;

}

/* ---------------------------------------------------- */
/* Bundle */
/* ---------------------------------------------------- */

.bundle-card{

background:#172554;

padding:18px;

border-radius:18px;

margin-bottom:20px;

}

/* ---------------------------------------------------- */
/* Comparison */
/* ---------------------------------------------------- */

.compare-card{

background:#111827;

padding:20px;

border-radius:18px;

margin-top:15px;

border:1px solid #374151;

}

/* ---------------------------------------------------- */
/* Buttons */
/* ---------------------------------------------------- */

.stButton>button{

background:#3B82F6;

color:white;

border:none;

border-radius:10px;

font-weight:bold;

padding:10px 18px;

}

.stButton>button:hover{

background:#2563EB;

}

/* ---------------------------------------------------- */
/* Chat Input */
/* ---------------------------------------------------- */

[data-testid="stChatInput"]{

position:fixed;

bottom:20px;

left:320px;

right:30px;

background:#0b1220;

padding-top:10px;

z-index:999;

}

</style>

""", unsafe_allow_html=True)
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding-top:10px;">
            <h2>🛍️ ShopWise AI</h2>
            <p style="color:#9CA3AF;">
                AI Shopping Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.shopping_context = {}

        st.session_state.conversation_complete = True

        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Try Asking")

    st.info("💻 Best laptop under ₹60,000")

    st.info("📱 Compare iPhone & Samsung")

    st.info("🎧 Best earbuds under ₹5,000")

    st.info("👟 Running shoes")

    st.info("👕 Black shirt for Goa")

    st.info("🎁 Gym shopping bundle")

    st.markdown("---")


    if st.session_state.shopping_context:

        st.markdown("### 🧠 Shopping Context")

        context = st.session_state.shopping_context

        if context.get("category"):

            st.write(
                f"📦 **Category:** {context['category']}"
            )

        if context.get("budget"):

            st.write(
                f"💰 **Budget:** ₹{context['budget']}"
            )

        if context.get("brand"):

            st.write(
                f"🏷 **Brand:** {context['brand']}"
            )

        if context.get("purpose"):

            st.write(
                f"🎯 **Purpose:** {context['purpose']}"
            )

        if context.get("occasion"):

            st.write(
                f"🎉 **Occasion:** {context['occasion']}"
            )


# ============================================================
# HEADER
# ============================================================

left, right = st.columns([8,2])

with left:

    st.markdown(
        """
<div class="main-title">

🛍️ ShopWise AI

</div>

<div class="sub-title">

Your Personal AI Shopping Assistant

</div>
""",
        unsafe_allow_html=True
    )

with right:

    if st.button("✨ New Chat"):

        st.session_state.messages = []

        st.session_state.shopping_context = {}

        st.session_state.conversation_complete = True

        st.rerun()


st.markdown("---")


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    with st.chat_message(
        "assistant",
        avatar="🛍️"
    ):

        st.markdown(
"""
## 👋 Hi! I'm **ShopWise AI**

I'm your intelligent shopping assistant.

I can help you with:

- 🛍️ Find the best products
- ⚖️ Compare products
- 🎁 Build shopping bundles
- 💰 Stay within your budget
- ⭐ Personalized recommendations

**What are you looking for today?**
"""
        )
# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = "👤" if message["role"] == "user" else "🛍️"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(

    "Ask ShopWise AI anything..."

)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "user",

            "content": user_query

        }

    )

    with st.chat_message(

        "user",

        avatar="👤"

    ):

        st.markdown(user_query)

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(

        "assistant",

        avatar="🛍️"

    ):

        placeholder = st.empty()

        with st.spinner(

            "Finding the best products..."

        ):

            try:

                response = requests.post(

                    API_URL,

                    json={

                        "query": user_query

                    },

                    timeout=120

                )

                response.raise_for_status()

                data = response.json()

                # --------------------------------------------
                # Extract API Response
                # --------------------------------------------

                answer = data.get(

                    "response",

                    "No response."

                )

                provider = data.get(

                    "provider",

                    "Unknown"

                )

                shopping_context = data.get(

                    "shopping_context",

                    {}

                )

                conversation_complete = data.get(

                    "conversation_complete",

                    True

                )

                products = data.get(

                    "products",

                    []

                )

                comparison = data.get(

                    "comparison",

                    None

                )

                bundle = data.get(

                    "bundle",

                    None

                )

                # --------------------------------------------
                # Save Conversation
                # --------------------------------------------

                st.session_state.shopping_context = (

                    shopping_context

                )

                st.session_state.conversation_complete = (

                    conversation_complete

                )

                # --------------------------------------------
                # Typing Animation
                # --------------------------------------------

                text = ""

                for letter in answer:

                    text += letter

                    placeholder.markdown(text)

                    time.sleep(0.003)

                st.caption(

                    f"⚡ Powered by {provider}"

                )

                # --------------------------------------------
                # Save Assistant Message
                # --------------------------------------------

                st.session_state.messages.append(

                    {

                        "role": "assistant",

                        "content": answer

                    }

                )
                # ============================================================
                # RECOMMENDED PRODUCTS
                # ============================================================

                if products:

                    st.markdown("---")

                    st.subheader("🛍️ Recommended Products")

                    for product in products:

                        # ----------------------------------------
                        # Support dict or Product object
                        # ----------------------------------------

                        if isinstance(product, dict):

                            name = product.get("product_name", "Unknown Product")
                            brand = product.get("brand", "Unknown")
                            price = product.get("price", "N/A")
                            rating = product.get("rating", 0)
                            explanation = product.get(
                                "explanation",
                                "Recommended for you."
                            )
                            image = product.get("image_url", "")
                            url = product.get("product_url", "")

                        else:

                            name = getattr(product, "product_name", "Unknown Product")
                            brand = getattr(product, "brand", "Unknown")
                            price = getattr(product, "price", "N/A")
                            rating = getattr(product, "rating", 0)
                            explanation = getattr(
                                product,
                                "explanation",
                                "Recommended for you."
                            )
                            image = getattr(product, "image_url", "")
                            url = getattr(product, "product_url", "")

                        st.markdown(
                            '<div class="product-card">',
                            unsafe_allow_html=True
                        )

                        left, right = st.columns([1, 2])

                        with left:

                            if image:

                                st.image(
                                    image,
                                    use_container_width=True
                                )

                            else:

                                st.image(
                                    "https://placehold.co/300x300?text=No+Image",
                                    use_container_width=True
                                )

                        with right:

                            st.markdown(
                                f"""
<div class="product-title">

{name}

</div>
""",
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f"""
<span class="brand">

🏷 {brand}

</span>
""",
                                unsafe_allow_html=True
                            )

                            st.markdown("<br>", unsafe_allow_html=True)

                            st.markdown(
                                f"""
<span class="price">

₹ {price}

</span>
""",
                                unsafe_allow_html=True
                            )

                            stars = "⭐" * int(float(rating))

                            st.markdown(
                                f"""
<div class="rating">

{stars} ({rating})

</div>
""",
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f"""
<div class="reason">

<b>🎯 Why Recommended</b>

<br><br>

{explanation}

</div>
""",
                                unsafe_allow_html=True
                            )

                            if url:

                                st.link_button(

                                    "🛒 View Product",

                                    url,

                                    use_container_width=True

                                )

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("<br>", unsafe_allow_html=True)

                                # ============================================================
                # PRODUCT COMPARISON
                # ============================================================

                if comparison:

                    st.markdown("---")

                    st.subheader("⚖️ Product Comparison")

                    if isinstance(comparison, dict):


                        st.json(comparison)

                    elif isinstance(comparison, list):

                        st.dataframe(
                            comparison,
                            use_container_width=True
                        )

                    else:

                        st.info(str(comparison))


                # ============================================================
                # SHOPPING BUNDLE
                # ============================================================

                if bundle:

                    st.markdown("---")

                    st.subheader("🎁 Recommended Bundle")

                    bundle_products = bundle.get(
                        "bundle_products",
                        []
                    )

                    for item in bundle_products:

                        if isinstance(item, dict):

                            image = item.get("image_url", "")
                            name = item.get("product_name", "")
                            brand = item.get("brand", "")
                            price = item.get("price", "")
                            url = item.get("product_url", "")

                        else:

                            image = getattr(item, "image_url", "")
                            name = getattr(item, "product_name", "")
                            brand = getattr(item, "brand", "")
                            price = getattr(item, "price", "")
                            url = getattr(item, "product_url", "")

                        st.markdown(
                            '<div class="bundle-card">',
                            unsafe_allow_html=True
                        )

                        c1, c2 = st.columns([1,3])

                        with c1:

                            if image:

                                st.image(
                                    image,
                                    use_container_width=True
                                )

                        with c2:

                            st.markdown(f"### {name}")

                            st.write(f"🏷 Brand : {brand}")

                            st.write(f"💰 Price : ₹{price}")

                            if url:

                                st.link_button(
                                    "🛒 View Product",
                                    url,
                                    use_container_width=True
                                )

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                                # ============================================================
                # NO RESULT FOUND
                # ============================================================

                if (

                    not products

                    and not comparison

                    and not bundle

                    and conversation_complete

                ):

                    st.info(

                        "🔍 No matching products found. Try another query."

                    )

            # ============================================================
            # API ERRORS
            # ============================================================

            except requests.exceptions.ConnectionError:

                st.error(

                    """
❌ Cannot connect to FastAPI.

Start the backend using:

uvicorn api.app:app --reload
"""

                )

            except requests.exceptions.Timeout:

                st.error(

                    "⏳ Request timed out. Please try again."

                )

            except requests.exceptions.HTTPError as e:

                st.error(

                    f"HTTP Error: {e}"

                )

            except Exception as e:

                st.exception(e)

# # ============================================================
# # FOOTER
# # ============================================================

# st.markdown("---")

# col1, col2 = st.columns(2)

# with col1:

#     st.caption("🛍️ ShopWise AI")


# with col2:

#     st.caption("🤖 AI Shopping Assistant")

# st.markdown(
# """
# <div style="text-align:center;
# padding:15px;
# color:#94A3B8;
# font-size:14px;">


# </div>
# """,
# unsafe_allow_html=True
# )
                    