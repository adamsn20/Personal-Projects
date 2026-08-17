import random
import streamlit as st

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Slay the Spire 2 Run Randomizer",
    page_icon="🗡️",
    layout="wide",
)

# Custom CSS for enhanced visual polish
st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Cinzel', 'Trajan Pro', serif;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #A8DADC;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .badge-ascension { background-color: #457B9D; color: white; }
    .badge-character { background-color: #2A9D8F; color: white; }
    .badge-pos { background-color: #388E3C; color: white; }
    .badge-neg { background-color: #D32F2F; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Constants & Data ---
ALL_CHARACTERS = ["Ironclad", "Silent", "Regent", "Necrobinder", "Defect"]

MUTUALLY_EXCLUSIVE_STARTERS = {"Draft", "Sealed Deck", "Insanity"}

POSITIVE_MODIFIERS = [
    "Draft",
    "Sealed Deck",
    "Hoarder",
    "Specialized",
    "Insanity",
    "All Star",
    "Flight",
    "Vintage",
    "Ironclad Cards",
    "Silent Cards",
    "Regent Cards",
    "Necrobinder Cards",
    "Defect Cards",
]

NEGATIVE_MODIFIERS = [
    "Deadly Events",
    "Cursed Run",
    "Big Game Hunter",
    "Midas",
    "Murderous",
    "Night Terrors",
    "Terminal",
]

# --- Helper Functions ---


def generate_positive_modifiers(count: int) -> list[str]:
    """Randomly selects positive modifiers ensuring at most 1 starter modifier."""
    if count == 0:
        return []

    # Decide whether to include one of [Draft, Sealed Deck, Insanity]
    available_starters = list(MUTUALLY_EXCLUSIVE_STARTERS)
    other_positives = [
        m for m in POSITIVE_MODIFIERS if m not in MUTUALLY_EXCLUSIVE_STARTERS
    ]

    # If count > len(other_positives) (i.e. > 10), we MUST include exactly 1 starter
    must_include_starter = count > len(other_positives)

    # Randomly pick if we include a starter (if not mandatory)
    include_starter = must_include_starter or (
        random.random() < (len(available_starters) / len(POSITIVE_MODIFIERS))
    )

    selected = []
    if include_starter:
        selected.append(random.choice(available_starters))

    needed = count - len(selected)
    selected.extend(random.sample(other_positives, needed))
    random.shuffle(selected)
    return selected


def reroll_positive_modifier(
    current_list: list[str], index_to_reroll: int
) -> list[str]:
    """Rerolls a single positive modifier while respecting exclusion rules."""
    old_mod = current_list[index_to_reroll]
    other_mods = [m for i, m in enumerate(current_list) if i != index_to_reroll]

    # Check if another modifier in the list already uses the exclusive starter pool
    has_starter = any(m in MUTUALLY_EXCLUSIVE_STARTERS for m in other_mods)

    # Available replacements cannot be already chosen
    candidates = [m for m in POSITIVE_MODIFIERS if m not in current_list]

    if has_starter:
        # If another starter exists, exclude all starters from candidate pool
        candidates = [
            c for c in candidates if c not in MUTUALLY_EXCLUSIVE_STARTERS
        ]

    if candidates:
        current_list[index_to_reroll] = random.choice(candidates)
    return current_list


# --- Session State Initialization ---
if "run_generated" not in st.session_state:
    st.session_state.run_generated = False
    st.session_state.ascension = None
    st.session_state.characters = []
    st.session_state.pos_modifiers = []
    st.session_state.neg_modifiers = []
    st.session_state.ascension_range = (0, 10)
    st.session_state.character_pool = ALL_CHARACTERS

# --- Header ---
st.markdown(
    "<h1 class='main-title'>🗡️ Slay the Spire 2 Run Randomizer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Configure your run parameters, randomize modifiers, ascension, and characters for solo or co-op spires.</p>",
    unsafe_allow_html=True,
)

# --- Configuration Layout (2 Columns) ---
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("### 🧗 Ascension Randomization")
    ascension_range = st.slider(
        "Ascension Level Range",
        min_value=0,
        max_value=10,
        value=(0, 10),
        step=1,
        help="Select the minimum and maximum Ascension level to randomize from (0–10).",
    )

    st.markdown("### 👥 Character Randomization")
    selected_char_pool = st.multiselect(
        "Available Characters",
        options=ALL_CHARACTERS,
        default=ALL_CHARACTERS,
        help="Choose which characters are allowed in the randomized roster.",
    )

    num_characters = st.number_input(
        "Number of Characters (for Co-op / Multiplayer)",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        help="Select how many characters to randomize (1 to 4). Duplicates are allowed.",
    )

with col2:
    st.markdown("### 🎲 Modifier Randomization")

    num_pos_modifiers = st.slider(
        "Positive Modifiers",
        min_value=0,
        max_value=11,
        value=0,
        step=1,
        help="Choose up to 11 positive modifiers. Max is 11 because 'Draft', 'Sealed Deck', and 'Insanity' are mutually exclusive.",
    )

    num_neg_modifiers = st.slider(
        "Negative Modifiers",
        min_value=0,
        max_value=7,
        value=0,
        step=1,
        help="Choose up to 7 negative modifiers without duplicates.",
    )

st.markdown("---")

# --- Randomize Button ---
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
with button_col2:
    if st.button(
        "🎲 Randomize Run", type="primary", use_container_width=True
    ):
        if not selected_char_pool:
            st.error(
                "⚠️ Please select at least one character in the character pool!"
            )
        else:
            # Store config in session state for rerolls
            st.session_state.ascension_range = ascension_range
            st.session_state.character_pool = selected_char_pool

            # Ascension
            st.session_state.ascension = random.randint(
                ascension_range[0], ascension_range[1]
            )

            # Characters (duplicates allowed)
            st.session_state.characters = [
                random.choice(selected_char_pool) for _ in range(num_characters)
            ]

            # Positive Modifiers (max 1 starter, no duplicates)
            st.session_state.pos_modifiers = generate_positive_modifiers(
                num_pos_modifiers
            )

            # Negative Modifiers (no duplicates)
            st.session_state.neg_modifiers = random.sample(
                NEGATIVE_MODIFIERS, num_neg_modifiers
            )

            st.session_state.run_generated = True

# --- Output Section ---
if st.session_state.run_generated:
    st.markdown("## 📜 Generated Run")

    out_col1, out_col2 = st.columns(2, gap="large")

    with out_col1:
        # --- Ascension Output ---
        st.markdown("#### 🧗 Ascension Level")
        asc_c1, asc_c2 = st.columns([4, 1])
        with asc_c1:
            st.info(f"**Ascension {st.session_state.ascension}**")
        with asc_c2:
            if st.button("🔄", key="reroll_ascension", help="Reroll Ascension"):
                st.session_state.ascension = random.randint(
                    st.session_state.ascension_range[0],
                    st.session_state.ascension_range[1],
                )
                st.rerun()

        # --- Character Output ---
        st.markdown("#### 👥 Character Selection")
        for idx, char in enumerate(st.session_state.characters):
            char_c1, char_c2 = st.columns([4, 1])
            with char_c1:
                st.success(f"**Player {idx + 1}:** {char}")
            with char_c2:
                if st.button(
                    "🔄",
                    key=f"reroll_char_{idx}",
                    help=f"Reroll Player {idx + 1}",
                ):
                    st.session_state.characters[idx] = random.choice(
                        st.session_state.character_pool
                    )
                    st.rerun()

    with out_col2:
        # --- Positive Modifiers Output ---
        st.markdown("#### 🟢 Positive Modifiers")
        if not st.session_state.pos_modifiers:
            st.caption("No positive modifiers selected.")
        else:
            for idx, mod in enumerate(st.session_state.pos_modifiers):
                pm_c1, pm_c2 = st.columns([4, 1])
                with pm_c1:
                    st.markdown(f"➕ **{mod}**")
                with pm_c2:
                    if st.button(
                        "🔄",
                        key=f"reroll_pos_{idx}",
                        help=f"Reroll {mod}",
                    ):
                        st.session_state.pos_modifiers = (
                            reroll_positive_modifier(
                                st.session_state.pos_modifiers, idx
                            )
                        )
                        st.rerun()

        # --- Negative Modifiers Output ---
        st.markdown("#### 🔴 Negative Modifiers")
        if not st.session_state.neg_modifiers:
            st.caption("No negative modifiers selected.")
        else:
            for idx, mod in enumerate(st.session_state.neg_modifiers):
                nm_c1, nm_c2 = st.columns([4, 1])
                with nm_c1:
                    st.markdown(f"➖ **{mod}**")
                with nm_c2:
                    if st.button(
                        "🔄",
                        key=f"reroll_neg_{idx}",
                        help=f"Reroll {mod}",
                    ):
                        available_neg = [
                            m
                            for m in NEGATIVE_MODIFIERS
                            if m not in st.session_state.neg_modifiers
                        ]
                        if available_neg:
                            st.session_state.neg_modifiers[idx] = random.choice(
                                available_neg
                            )
                        st.rerun()
