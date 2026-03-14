from __future__ import annotations

from pathlib import Path

import streamlit as st

from classify_tumor import DEFAULT_DATA_DIR, build_model, classify_tumor


@st.cache_resource
def get_model(data_dir: str):
    return build_model(Path(data_dir))


def main() -> None:
    st.set_page_config(page_title="Tumor Image Classifier", page_icon="🩺", layout="centered")
    st.title("Tumor Image Classifier")
    st.caption("上传图像并填写相关指标后，系统将输出良性/恶性判断和置信度。")

    data_dir = DEFAULT_DATA_DIR

    if not data_dir.exists():
        st.error("未找到 `data` 目录。请在项目中准备 `data/good` 和 `data/bad` 参考图像后再部署。")
        st.stop()

    try:
        model = get_model(str(data_dir))
    except Exception as exc:
        st.error(f"模型初始化失败：{exc}")
        st.stop()

    uploaded_file = st.file_uploader("上传肿瘤图像", type=["png", "jpg", "jpeg", "bmp"])
    ca19_9 = st.number_input("CA19-9 指标", min_value=0.0, value=0.0, step=1.0)
    tumor_size = st.number_input("肿瘤大小(最大宽度)", min_value=0.0, value=0.0, step=0.1)

    if uploaded_file is not None:
        st.image(uploaded_file, caption="待分析图像", use_container_width=True)

    if st.button("开始分析", type="primary", disabled=uploaded_file is None):
        try:
            result = classify_tumor(
                uploaded_file.getvalue(),
                ca19_9=ca19_9,
                tumor_size=tumor_size,
                model=model,
            )
        except Exception as exc:
            st.error(f"分析失败：{exc}")
            st.stop()

        if result["label"] == "bad":
            st.error(f"预测结果：{result['label_text']}")
        else:
            st.success(f"预测结果：{result['label_text']}")

        st.metric("置信度", result["confidence_percent"])
        st.progress(int(round(result["confidence"] * 100)))

    st.divider()
    st.warning(
        "免责声明：本网站提供的临床数据和预测结果仅供科研参考，不能作为临床诊断或治疗决策的依据。"
        "数据来源可能存在选择性偏倚，实际应用时请结合专业医学判断。"
        "使用者需自行承担风险，本平台不承担因信息使用引发的任何责任。"
    )


if __name__ == "__main__":
    main()
