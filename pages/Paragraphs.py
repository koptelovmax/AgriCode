# Combined version (segment construction + plain text)
#
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import base64

st.set_page_config(
    page_title="AgriCode",
    page_icon="icon.png"
)

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

@st.cache_resource
def load_model7():
    return BertForSequenceClassification.from_pretrained("koptelovmax/bert-base-ecozept")

@st.cache_resource
def load_model7plus():
    return BertForSequenceClassification.from_pretrained("koptelovmax/bert-ecozept-augmented")

@st.cache_resource
def load_model15():
    return BertForSequenceClassification.from_pretrained("koptelovmax/bert-base-ecozept-15")

def set_header():
    LOGO_IMAGE = "agriloop-logo.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #f9a01b !important;
            padding-left: 10px !important;
        }
        .logo-img {
            float:right;
            width: 28%;
            height: 28%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">AgriCode <span style="color:grey;">paragraphs</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Define tokenizer:
tokenizer = load_tokenizer()

#st.header('Code prediction tool: paragraphs', divider='blue')
set_header()


tab1, tab2 = st.tabs(["Segment constructor", "Plain text"])

with tab1:
    # Segment constructor:
    questionnaire = st.selectbox(
    "Select questionnaire:",
    ('Questionnaire for arable farmers/ horticulture businesses/ plant production / food processors (= suppliers of residual streams)',
     'Questionnaire for arable farmers/ horticulture businesses as users of the end-product PHA applications (mulch films etc.)',
     'Questionnaire for traders, processors, converters of the end-product PHA',
     'Questionnaire for feed traders and processors of the end-product extracted and microbial proteins'),
    )
    
    if questionnaire == "Questionnaire for arable farmers/ horticulture businesses/ plant production / food processors (= suppliers of residual streams)":
        block_name = st.selectbox(
        "Select block:",
        ('Block: Introduction and ice breaker',
         'Block: Strengths and challenges',
         'Block: Current structures',
         'Block: Expectations',
         'Block: Outlook and perspectives',
         'Last page for all of the interview guidelines above'),
        )
    elif questionnaire == "Questionnaire for arable farmers/ horticulture businesses as users of the end-product PHA applications (mulch films etc.)":
        block_name = st.selectbox(
        "Select block:",
        ('Block: Introduction and ice breaker',
         'Block: Market opportunities',
         'Block: Stakeholders’ expectations',
         'Block: Limits and barriers',
         'Block: Outlook and perspectives',
         'Last page for all of the interview guidelines above'),
        )
    elif questionnaire == "Questionnaire for traders, processors, converters of the end-product PHA":
        block_name = st.selectbox(
        "Select block:",
        ('Block: Introduction and ice breaker',
         'Block: Market opportunities',
         'Block: Stakeholders’ expectations',
         'Block: Limits and barriers',
         'Last page for all of the interview guidelines above'),
        )
    elif questionnaire == "Questionnaire for feed traders and processors of the end-product extracted and microbial proteins":
        block_name = st.selectbox(
        "Select block:",
        ('Block: Introduction and ice breaker',
         'Block: Market opportunities',
         'Block: Stakeholders’ expectations',
         'Block: Limits and barriers',
         'Last page for all of the interview guidelines above'),
        )
        
    if block_name == "Block: Introduction and ice breaker":
        question = st.selectbox(
        "Select question:",
        ('To begin with, please briefly describe your company and your position within the company',
         'Which type of residue stream (apple, tomato, grape, potatoes, or brewer’s grains) do you deal with in your company?',
         'Which products made of plastic (mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags) do you use in your farm/ business?',
         'What final material/products (mulch films, horticultural pots, ropes, growth protectors, greenhouses, nets, PHA growing/seeding bags) do you manufacture or market in your company?'),
        )
    elif block_name == "Block: Market opportunities":
        question = st.selectbox(
        "Select question:",
        ('For which agricultural applications is PHA not suitable? Why? Which functions are missing?',
         'How do you assess the market potential of extracted/microbial proteins?'
         'How do you assess the market potential of mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from PHA? How do you assess it compared to conventional materials?',
         'In how far are bio-based and bio-degradable  mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues a better solution than products from petrochemicals for you?',
         'In how far do you think that mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags are the best application of PHA in agriculture?',
         'In which markets can these proteins be applied?',
         'To which extent could extracted/ microbial proteins fulfill the same purpose as conventional proteins?',
         'What applications are inappropriate for the use of extracted/microbial proteins?',
         'What do you consider to be the best use of extracted/ microbial proteins regarding their functionality (e.g., feed for aquaculture)?',
         'What functions might extracted/microbial proteins not perform compared to conventional proteins?',
         'What other applications (besides animal feed) of extracted/microbial proteins do you find attractive in terms of functionality?',
         'Which applications are inappropriate for the use of extracted/ microbial proteins?',
         'Which disadvantages do they have?',
         'Which factors determine market potential of mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags of PHA predominantly?',
         'Which further advantages do bio-based and bio-degradable  mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues have for you?',
         'Which other applications (besides feedstuff) of extracted/ microbial proteins are attractive to you in terms of functionality?',
         'Which other applications of PHA are attractive in terms of functionality (e.g., mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags, etc.)?'),
        )
    elif block_name == "Block: Stakeholders’ expectations":
        question = st.selectbox(
        "Select question:",
        ('How do you assess price development of PHA applications for agriculture, in this case for mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags? On which factors does the price development depend?',
         'How do you assess the qualities of extracted/microbial proteins compared to conventional proteins?',
         'In case you produce composite products: Which degree of purity do clients ask for mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags? Does it have to consist from 100% PHA? Could it partwise be supplemented with, e.g., cellulose?',
         'In how far is contamination of the material an issue?',
         'Under which circumstances would you be willing to pay a higher price for bio-based and bio-degradable mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues than for conventional materials?',
         'What are your expectations regarding the qualities of agro-residues as a raw material for extracted/microbial protein processing?'
         'What are your quality expectations for extracted or microbial proteins?',
         'Which degree of purity do you need for mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags? Does it have to consist from 100% PHA as material? Could it partwise be supplemented with, e.g., cellulose?',
         'Which expectations regarding the qualities of agri-residues as feedstock for processing of extracted/ microbial proteins do you have?',
         'Which expectations to extracted/ microbial proteins do you have regarding quality, e.g. regarding: anti-nutritional factors',
         'Which expectations to extracted/ microbial proteins do you have regarding quality, e.g. regarding: contamination',
         'Which expectations to extracted/ microbial proteins do you have regarding quality, e.g. regarding: microbial risks',
         'Which expectations to extracted/ microbial proteins do you have regarding quality, e.g. regarding: Toxicity',
         'Which expectations to mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags of PHA do you have regarding its quality?',
         'Which functions do bio-based and bio-degradable mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues  have to fulfill for you?',
         'Which functions does PHA as material has to fulfill for use in agricultural applications like mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags?'),
        )
    elif block_name == "Block: Limits and barriers":
        question = st.selectbox(
        "Select question:",
        ('How could these challenges be overcome?',
         'How do you assess the costs of conventional proteins compared to extracted/ microbial proteins?',
         'How do you assess the future development of prices for extracted/ microbial proteins?',
         'How do you assess the legislation regarding approval of extracted/ microbial proteins as feed (e.g., preferential treatment of extracted/ microbial as feed, impediment of extracted/ microbial as feed, etc.)?',
         'How do you assess the scalability of extracted/ microbial proteins from agricultural residues?',
         'How do you rate the costs of conventional proteins compared to extracted/microbial proteins',
         'How does scalability limit today’s market success of extracted/ microbial proteins?',
         'How does the current commercial success of extracted/microbial proteins limit scalability'
         'How does the geographic distribution of agricultural residues influence the production of extracted/microbial proteins?',
         'How fast could scalability of PHA applications from agricultural residues like mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags take place in your opinion?',
         'On which factors does the price of extracted/ microbial proteins depend?',
         'Please describe the challenges for the use of bio-based and bio-degradable mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues for you.',
         'Please describe the challenges regarding the application of mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from PHA',
         'What does scalability depend on?',
         'Which other factors limit the market potential of extracted/microbial proteins, e.g.: availability of extracted/microbial proteins as feedstuff',
         'Which other factors limit the market potential of extracted/microbial proteins, e.g.: consumer acceptance',
         'Which other factors limit the market potential of extracted/microbial proteins, e.g.: Legislation',
         'Which other factors limit the market potential of extracted/microbial proteins, e.g.: price',
         'Which other factors limit the market potential of extracted/microbial proteins, e.g.: scalability, etc.?'),
        )
    elif block_name == "Block: Strengths and challenges":
        question = st.selectbox(
        "Select question:",
        ('Are you aware of other alternative valorization pathways that might be better than the current one? Yes – No if Yes:  Which ones?',
         'How could these challenges be overcome?',
         'How do you currently valorize your agricultural by-products of apple-/tomato-/grape-/potato processing/brewer’s grains?',
         'How satisfied are you with your current valorization of apple-/tomato-/grape-/potato processing/brewer’s grains? Why?',
         'Please describe your challenges regarding the valorization of your agricultural by-products from apple-/tomato-/grape-/potato processing /brewer’s grains'
         'When you provide your agricultural by-products for this valorization pathway, how much do you get paid? Who organizes the transport of the by-products? Who pays for the transportation of the by-products?',
         'Which are the strengths and advantages in the current valorization of by-products of apple-/tomato-/grape-/potato processing/brewer’s grains?'),
        )    
    elif block_name == "Block: Current structures":
        question = st.selectbox(
        "Select question:",
        ('Are you aware of other alternative valorisation pathways that might be better than the current one? Yes – No if Yes:  Which ones?',
         'How satisfied are you with your current valorization of apple-/tomato-/grape-/potato processing/brewer’s grains? Why?',
         'How satisfied are you with your current valorization of apple-/tomato-/grape-/potato processing/brewer’s grains?'
         'When you provide your agricultural by-products for this valorization pathway, how much do you get paid? Who organizes the transport of the by-products? Who pays for the transportation of the by-products?'),
        )    
    elif block_name == "Block: Expectations":
        question = st.selectbox(
        "Select question:",
        ('Do you observe any new interesting solutions in the area of valorisation of by-products from apple-/tomato-/grape-/potato processing /brewer’s grains in your professional environment?',
         'In your opinion, where is the development going regarding the use of the by-products from apple-/tomato-/grape-/potato processing /brewer’s grains?',
         'What are your expectations regarding the valorization of by-products from apple-/tomato-/grape-/potato processing /brewer’s grains?'
         'What kind of support would you need for this ideal valorization of the by-products of apple-/tomato-/grape-/potato processing /brewer’s grains?',
         'Which would be your ideal solution for the valorization of your by-products of apple-/tomato-/grape-/potato processing /brewer’s grains? Could you explain, why?'),
        )    
    elif block_name == "Block: Outlook and perspectives":
        question = st.selectbox(
        "Select question:",
        ('Are there aspects about the topic of the interview that are relevant for you but that we did not consider in our interview? Could you specify them please?',
         'Are you interested in receiving a summary of the results of this study? This document will of course not disclose any confidential information. If yes, I would need your e-mail address, please:',
         'Do you know further experts about the topic of the interview who would be interested to participate in the survey?',
         'Do you observe any new interesting solutions in the area of bio-based and bio-degradable mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues in your personal surrounding?',
         'Do you observe any new interesting solutions in the area of valorization of by-products from apple-/tomato-/grape-/potato processing /brewer’s grains in your professional environment?',
         'In your opinion, where is the development going regarding the use of the by-products from apple-/tomato-/grape-/potato processing /brewer’s grains?',
         'In your opinion, where is the development going with regard to the application of bio-based and bio-degradable mulch films, horticultural pots, twines, growth protectors, greenhouses, nets, grow/seedling bags from agri-residues?'),
        )    
    elif block_name == "Last page for all of the interview guidelines above":
        question = st.selectbox(
        "Select question:",
        ('Are there aspects about the topic of the interview that are relevant for you but that we did not consider in our interview? Could you specify them please?',
         'Are you interested in receiving a summary of the results of this study? This document will of course not disclose any confidential information. If yes, I would need your e-mail address, please:',
         'Do you know further experts about the topic of the interview who would be interested to participate in the survey?'),
        )    
    
    paragraphs = st.text_area(
    "Your paragraph:",
    "Yes, of course! We are producing all sorts of different potato convenient products such as croquettes, french fries, “Croustille”, and a range of different stuffed speciality potato products. But all our products are frozen convenient potato products.  I am working for the department of sustainability, energy and environment of our company and I am responsible for issues related to sustainability, energy, waste, and environment. I am responsible for the valorisation of our potato wastes through our biogas plant.\n\n"
    "We are very happy with our current valorisation process. The valorisation is connected to a decent amount of effort but it is working really well. It is very important for us to get rid of the residues from our food production operation as quick as possible and we are more or less able to feed the residues into the biogas plant the moment the waste accumulates. The waste from the processing has to be removed continuously and this works very well. The residues of the biogas plant also need to be disposed every day and this works really well as well. Our cooperation with the logistics company works flawless and this is really important to us as we are not able to store the residues. They would need to be stored inside and this would be very costly as the hygiene would be an issue.",
    height=160,
    )
    
    text_to_classify = questionnaire+"\n\n"+block_name+"\n\n"+question+"\n\n"+paragraphs
    
with tab2:
    # Plain text:
    segment_text = st.text_area(
    "Text to classify:",
    text_to_classify,
    height=320,
    )
    
    text_to_classify = segment_text

mode = st.radio(
"Select classifier:",
["7 classes", "15 classes"],
)

# Select 7 or 15 class setting:
if mode == "7 classes":
    col_1, col_2 = st.columns([1,1], vertical_alignment="top")

    with col_1:
        # Select considering titles:
        titles_flag = st.checkbox("Take titles into account", value=True)
    with col_2:
        # Select augmented classifier:
        augm = st.checkbox("Improved classifier (data augmentation)", value=True)
    
    # Load model:
    if augm:
        model = load_model7plus()
        model.to('cpu')
    else:
        model = load_model7()
        model.to('cpu')
    
    if augm:
        if titles_flag:
            st.write("You have selected the 7-class setting with an improved classifier, taking into account titles.")
        else:
            st.write("You have selected the 7-class setting with an improved classifier.")
    else:
        if titles_flag:
            st.write("You have selected the 7-class setting, taking into account titles.")
        else:
            st.write("You have selected the 7-class setting.")
else:
    # Select considering titles:
    titles_flag = st.checkbox("Take titles into account", value=True)
        
    # Load model:
    model = load_model15()
    model.to('cpu')
    if titles_flag:
        st.write("You have selected the 15-class setting, taking into account titles.")
    else:
        st.write("You have selected the 15-class setting.")

def prediction(segment_text):
    test_ids = []
    test_attention_mask = []
    
    # Apply the tokenizer
    encoding = tokenizer(segment_text, padding="longest", return_tensors="pt")
    
    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)
    
    # Forward pass, calculate logit predictions
    with torch.no_grad():
      output = model(test_ids.to('cpu'), token_type_ids = None, attention_mask = test_attention_mask.to('cpu'))
    
    return np.argmax(output.logits.cpu().numpy()).flatten().item()

def main():
       
    if st.button('Predict'):
        # Separate paragraphs:
        if titles_flag:
            titles = "\n\n".join(text_to_classify.split("\n\n")[:3])
            segment_paras = text_to_classify.split("\n\n")[3:]
        else:
            segment_paras = text_to_classify.split("\n\n")
        
        col1, col2 = st.columns([2,1], vertical_alignment="top")

        with col1:
            # Output result of prediction:
            st.markdown("**Result or prediction:**")
            
            paras_highlighted = ""
            
            if mode == "7 classes":
                for i in range(len(segment_paras)):
                    # Predict label for each paragraph:
                    if titles_flag:
                        code_id = prediction(titles+"\n\n"+segment_paras[i])
                    else:
                        code_id = prediction(segment_paras[i])
                    
                    # Highlight each paragraph w.r.t. legend and predicted code:
                    if code_id == 1:
                        paras_highlighted += ":red-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 2:
                        paras_highlighted += ":blue-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 3:
                        paras_highlighted += ":green-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 4:
                        paras_highlighted += ":violet-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 5:
                        paras_highlighted += ":orange-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 6:
                        paras_highlighted += ":grey-background["+segment_paras[i]+"]\n\n"
                    else:
                        paras_highlighted += segment_paras[i]+"\n\n"
            else:
                for i in range(len(segment_paras)):
                    # Predict label for each sentence:
                    if titles_flag:
                        code_id = prediction(titles+"\n\n"+segment_paras[i])
                    else:
                        code_id = prediction(segment_paras[i])
                    
                    # Highlight each sentence w.r.t. legend and predicted code:
                    if code_id == 1:
                        paras_highlighted += ":red-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 2:
                        paras_highlighted += ":blue-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 3:
                        paras_highlighted += ":green-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 4:
                        paras_highlighted += ":violet-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 5:
                        paras_highlighted += ":orange-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 6:
                        paras_highlighted += ":grey-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 7:
                        paras_highlighted += ":rainbow-background["+segment_paras[i]+"]\n\n"
                    elif code_id == 8:
                        paras_highlighted += ":red["+segment_paras[i]+"]\n\n"
                    elif code_id == 9:
                        paras_highlighted += ":blue["+segment_paras[i]+"]\n\n"
                    elif code_id == 10:
                        paras_highlighted += ":green["+segment_paras[i]+"]\n\n"
                    elif code_id == 11:
                        paras_highlighted += ":violet["+segment_paras[i]+"]\n\n"
                    elif code_id == 12:
                        paras_highlighted += ":orange["+segment_paras[i]+"]\n\n"
                    elif code_id == 13:
                        paras_highlighted += ":grey["+segment_paras[i]+"]\n\n"
                    elif code_id == 14:
                        paras_highlighted += ":rainbow["+segment_paras[i]+"]\n\n"
                    else:
                        paras_highlighted += segment_paras[i]+"\n\n"
            
            # Output highlighted paragraphs:
            st.markdown(paras_highlighted)
        
        with col2:
            # Legend:
            if mode == "7 classes":
                st.markdown('''**Legend:**  
                        Other (not pertinent)  
                        :red-background[limitations and barriers]  
                        :blue-background[Stakeholders’ expectations]  
                        :green-background[market opportunities]  
                        :violet-background[valorization]  
                        :orange-background[company+experts]  
                        :grey-background[type of stream]''')
            else:
                st.markdown('''**Legend:**  
                        Other (not pertinent)  
                        :red-background[Stakeholders’ expectations > valorization/ PHA-applications]  
                        :blue-background[limitations and barriers > valorization /PHA-applications]  
                        :green-background[market opportunities > PHA MO]  
                        :violet-background[market opportunities > PHA-Applications MO]  
                        :orange-background[valorization > current structures]  
                        :grey-background[company+experts]  
                        :rainbow-background[limitations and barriers > Main issues and challenges for extracted/microbial protein]  
                        :red[type of stream]  
                        :blue[Stakeholders’ expectations > PHA expectation]  
                        :green[limitations and barriers > Main issues and challenges for PHA]  
                        :violet[market opportunities > MP MO]  
                        :orange[Stakeholders’ expectations > MP]  
                        :grey[valorization > satisfaction]  
                        :rainbow[valorization > advantages]''')
        #st.write("Predicted code: ", "**"+pred_label+"**")
    
if __name__ == "__main__":
    main()
        
    
