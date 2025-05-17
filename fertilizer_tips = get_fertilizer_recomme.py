fertilizer_tips = get_fertilizer_recommendations(N, P, K)
for tip in fertilizer_tips:
    st.info(f"{tip['message']}\n- Application: {tip['application']}\n- Timing: {tip['timing']}\n- Note: {tip['precaution']}")