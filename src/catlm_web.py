from __future__ import annotations

import os
import streamlit as st

# ---- import your CATLM code ----
# from catlm import CATLMAgent, CatProfile, Action

def attach_bank_if_any(cat, bank_path: str):
    if bank_path and os.path.exists(bank_path):
        try:
            from dialogue_bank import attach_dialogue_bank, sample_dialogue
            attach_dialogue_bank(cat, bank_path)
            cat.dialogue = lambda: sample_dialogue(cat)
            return True
        except Exception:
            return False
    return False

def init_agent(seed: int, bank_path: str):
    cat = CATLMAgent(
        profile=CatProfile(name="Nabi", activity=4, sociability=5, appetite=3, cowardice=4),
        rng_seed=seed,
    )
    ok = attach_bank_if_any(cat, bank_path)
    return cat, ok

def main():
    st.set_page_config(page_title="CATLM Demo", layout="wide")
    st.title("CATLM Demo (Python Wrapper)")

    with st.sidebar:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42)
        bank_path = st.text_input("Dialogue bank path", value="dialogue_bank_512.json")
        if st.button("Reset Agent"):
            st.session_state.pop("cat", None)

    if "cat" not in st.session_state:
        cat, ok = init_agent(int(seed), bank_path)
        st.session_state.cat = cat
        st.session_state.bank_ok = ok

    cat = st.session_state.cat

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Controls")
        action = st.selectbox("Action", [a.value for a in Action], index=4)  # default pet-ish
        if st.button("Apply Action"):
            cat.apply_action(Action(action))
            cat.tick(hours=1)

        hours = st.slider("Tick hours", 1, 6, 2)
        if st.button("Tick"):
            cat.tick(hours=int(hours))

        st.caption("Tip: Apply Action 후 Tick으로 리듬을 만들어 보세요.")

    with col2:
        st.subheader("Readout")
        Ct = cat.crisis_score()
        theta = cat.crisis_threshold()
        st.metric("Crisis Score (Ct)", f"{Ct:.3f}")
        st.metric("Threshold (theta)", f"{theta:.3f}")
        st.metric("Stage", cat.crisis_stage())

        if hasattr(cat, "care_alpha"):
            st.metric("care_alpha (α proxy)", f"{cat.care_alpha:.3f}")
        if hasattr(cat, "capacity"):
            st.metric("capacity", f"{cat.capacity:.3f}")
        if hasattr(cat, "mode"):
            st.write("mode:", str(cat.mode))

        st.write("dialogue:", cat.dialogue())

    st.divider()
    st.subheader("State Vector")
    s = cat.state.values
    st.json({k.value: int(v) for k, v in s.items()})

    st.caption("Bank attached: " + ("✅" if st.session_state.get("bank_ok") else "❌ (built-in dialogue fallback)"))

if __name__ == "__main__":
    main()