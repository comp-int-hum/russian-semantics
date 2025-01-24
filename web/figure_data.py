import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json


st.title("Simple Data Dashboard")

visualize_option = st.radio("What information do you want to look at: ", ["None Placeholder", "Vocabulary Distribution Changes", "Evolution Timeline"])

if visualize_option == "Vocabulary Distribution Changes":

    file_dir = '/home/zxia15/russian-semantics/work/topic_divergence_data.csv'
    df = pd.read_csv(file_dir)

    # # let's restore the numpy array!
    data_values = df.drop(columns=["divergence", " year"], axis=1).values
    divergence_arr = data_values.reshape(3, 4, 20)

    time_intervals = ["1850-60", "1860-70", "1870-80", "1880-1890"]
    topic_display_option = st.radio("Topic Display: ", ["single topic", "cross-topic comparison"])

    if topic_display_option == "single topic":

        # after doing that, we can have a data column to choose the topics and choose the divergence wanted
        topic_idx = st.number_input(
            "Please input topic number", value=None, step=1)
        if topic_idx is None or topic_idx < 0 or topic_idx >= 20:
            st.write("Please input a valid topic number")
        else:

            topic_idx = int(topic_idx)
            divergence_selection = st.multiselect(
                "Select the divergence you want to display",
                ["KL divergence forward", "KL divergence backward", "JS divergence"],
                ["KL divergence forward", "KL divergence backward", "JS divergence"]
            )
    
            fig, ax = plt.subplots(figsize=(10, 6))

            if "KL divergence forward" in divergence_selection or "KL divergence backward" in divergence_selection:
                scale_flag = st.checkbox("normalize KL divergence")
                scaled_divergence = [(divergence_arr[fb_idx, :, topic_idx] - divergence_arr[fb_idx, :, topic_idx].min()) / (divergence_arr[fb_idx, :, topic_idx].max() - divergence_arr[fb_idx, :, topic_idx].min() + 1e-10) for topic_idx in range(20) for fb_idx in range(2)]
        
            if "KL divergence forward" in divergence_selection:
                ax.plot(time_intervals, scaled_divergence[topic_idx * 2] if scale_flag else divergence_arr[0, :, topic_idx], label=f"Topic {topic_idx} DL Forward", linestyle='-', linewidth=1.5)
            if "KL divergence backward" in divergence_selection:
                ax.plot(time_intervals, scaled_divergence[topic_idx * 2 + 1] if scale_flag else divergence_arr[1, :, topic_idx], label=f"Topic {topic_idx} DL Backward", linestyle=':', linewidth=1.5)
            if "JS divergence" in divergence_selection:
                ax.plot(time_intervals, divergence_arr[2, :, topic_idx], label=f"Topic {topic_idx} JS", linestyle='-.', linewidth=1.5)
            
            ax.set_xlabel("Time Interval")
            ax.set_ylabel("Divergence Value")
            ax.set_title(f"Topic {topic_idx} KL & JS Divergence")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
    
    elif topic_display_option == "cross-topic comparison":
        topic_idxes = st.text_input("Input all topic of interest, separated by a comma and with no space in between:", "")

        proceed_flag = False
        
        try:
            topic_arr = topic_idxes.split(",")
            topic_arr = [int(topic_idx) for topic_idx in topic_arr if int(topic_idx) >= 0 and int(topic_idx) < 20]
            proceed_flag = True
        except Exception as e:
            st.write(f"encountering error when trying to parse the topics: {str(e)}")
        
        if proceed_flag is True and len(topic_arr) > 0:

            divergence_selection = st.multiselect(
                "Select the divergence you want to display",
                ["KL divergence forward", "KL divergence backward", "JS divergence"],
                ["KL divergence forward", "KL divergence backward", "JS divergence"]
            )
    
            fig, ax = plt.subplots(figsize=(10, 6))
            cmap = plt.get_cmap('tab10')  # Just use one argument here
            colors = [cmap(i % cmap.N) for i in range(20)]
            if "KL divergence forward" in divergence_selection or "KL divergence backward" in divergence_selection:
                scale_flag = st.checkbox("normalize KL divergence")
                scaled_divergence = [(divergence_arr[fb_idx, :, topic_idx] - divergence_arr[fb_idx, :, topic_idx].min()) / (divergence_arr[fb_idx, :, topic_idx].max() - divergence_arr[fb_idx, :, topic_idx].min() + 1e-10) for topic_idx in range(20) for fb_idx in range(2)]

            for topic_idx in topic_arr:
        
                if "KL divergence forward" in divergence_selection:
                    ax.plot(time_intervals, scaled_divergence[topic_idx * 2] if scale_flag else divergence_arr[0, :, topic_idx], label=f"Topic {topic_idx} DL Forward", linestyle='-', linewidth=1.5, color=colors[topic_idx])
                if "KL divergence backward" in divergence_selection:
                    ax.plot(time_intervals, scaled_divergence[topic_idx * 2 + 1] if scale_flag else divergence_arr[1, :, topic_idx], label=f"Topic {topic_idx} DL Backward", linestyle=':', linewidth=1.5, color=colors[topic_idx])
                if "JS divergence" in divergence_selection:
                    ax.plot(time_intervals, divergence_arr[2, :, topic_idx], label=f"Topic {topic_idx} JS", linestyle='-.', linewidth=1.5, color=colors[topic_idx])
            
            ax.set_xlabel("Time Interval")
            ax.set_ylabel("Divergence Value")
            ax.set_title(f"Topic {topic_idxes} KL & JS Divergence")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

elif visualize_option == "Evolution Timeline":

    with open('/home/zxia15/russian-semantics/work/all_topic_notes.json', 'r', encoding='utf-8') as f:
        all_topic_notes = json.load(f)

    topic_spec = [
        "different modalities / causality; logical quantification",
        "measurement: monetary: military -> relation -> objects of measurement -> numerical"
        "legal and letters",
        "livelihood: potentially gentry -> village",
        "scientific cataloging: linguistics / text -> religion / monarch -> political parties",
        "religious authority -> state authority -> monetary units and numbers",
        "logical connection -> spacial relation -> body parts",
        "unclear",
        "numerical system -> statehood, military",
        "peasantry, authority -> political parties and other terminologies -> unclear (last 2)",
        "unclear",
        "unclear",
        "speech: body parts -> addresses -> more formal speaker and political speech",
        "unclear -> history, language -> state, military -> unclear -> religion",
        "unclear",
        "personal info (appearance, name) -> location -> time -> relation (spatial, requential, logical, numerical)",
        "state -> religion -> unclear (last 3)",
        "unclear -> speech, time -> name, appearance -> linguistic, Russia -> publication",
        "unclear",
        "quantity, numerical system -> monetary and military -> statehood"
    ]

    filtered_topic = [f"Topic #{topic_idx}: {topic_text}" for topic_idx, topic_text in enumerate(topic_spec) if topic_text != "unclear"]
    topic_choice = st.selectbox(
        "Please input topic number", filtered_topic, index=None)
    if topic_choice is None:
        st.write("Please input a valid topic number")
    else:
        topic_idx = int((topic_choice.split(": ")[0]).split("#")[-1])
        txt = st.text_area(
        "Notes on this topic",
        all_topic_notes[str(topic_idx)],
        height=50
        )
        
        if st.button("Save", type="primary"):
        
            if len(txt) > 0:
                all_topic_notes[str(topic_idx)] = txt
                with open('/home/zxia15/russian-semantics/work/all_topic_notes.json', 'w', encoding='utf-8') as f:
                    json.dump(all_topic_notes, f, indent=4)

        # visualize_option = st.radio("Choose the statistics: ", ["Proportion within a single Topic", "Distribution of topic within all appearance"])
        
        # key_word = "dist" if visualize_option == "Distribution of topic within all appearance" else "prop"

        # author_image_dir = f"/home/zxia15/russian-semantics/images/topic#{topic_idx}_top_{key_word}_authors.png"
        # work_image_dir = f"/home/zxia15/russian-semantics/images/topic#{topic_idx}_top_{key_word}_works.png"

        st.image(f"/home/zxia15/russian-semantics/images/topic#{topic_idx}_combined.jpg")