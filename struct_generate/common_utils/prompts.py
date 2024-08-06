import json


TOPIC_PROPOSE_PROMPT = \
"""### Instruction
I need to construct a comprehensive benchmark to evaluate LLMs' knowledge about a specific objective.
For a given assessment objective, you must propose {topic_num} most important knowledge points that must be evaluated about this objective.
The reponse must follow the provided json format: [{{"name": <name of knowledge point>, "description": <description of knowledge point>}}, ...]
You will be penalized for proposing irrelevant or unimportant knowledge points.

### Example
Assessment objective: 2022 FIFA World Cup
5 most important knowledge points about 2022 FIFA World Cup:
```json
[
    {{"name": "Host country of 2022 FIFA World Cup", "description": "Knowledge about Qatar as the host country, including its selection process and preparations for the World Cup."}},
    {{"name": "Teams participating in 2022 FIFA World Cup", "description": "Information about the national teams that qualified and participated in the tournament."}},
    {{"name": "Key players of 2022 FIFA World Cup", "description": "Information about prominent players who played significant roles in the tournament."}},
    {{"name: "Knockout stage matches in 2022 FIFA World Cup", "description": "Details of the knockout stages, including the round of 16, quarter-finals, semi-finals, and finals."}},
    {{"name": "Group stage matches in 2022 FIFA World Cup", "description": "Information on how the group stage was organized and the outcomes of these matches."}}
]
```

### Input
Assessment objective: {objective}
{topic_num} most important knowledge points about {objective}:
"""

CONCEPT_PROPOSE_PROMPT = \
"""### Instruction
I need to construct a comprehensive benchmark to evaluate LLMs' knowledge about a specific objective.
For a given assessment objective, you must propose {concept_num} most critical concepts that must be understood about this objective.
The reponse must follow the provided json format: [{{"name": <name of concept>, "description": <description of concept>}}, ...]
You will be penalized for proposing irrelevant or unimportant concepts.

### Example
Assessment objective: 2022 FIFA World Cup
5 most critical concepts about 2022 FIFA World Cup:
```json
[
    {{"name": "Qatar", "description": "The host country of 2022 FIFA World Cup"}},
    {{"name": "Lionel Messi", "description": "An Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. "}},
    {{"name": "France National Team", "description": "The defending champions from the previous World Cup."}},
    {{"name: "Lusail Iconic Stadium", "description": "One of the key stadiums in 2022 FIFA World Cup, especially for the final match."}},
    {{"name": "Kylian Mbappé", "description": "A French professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the France national team."}}
]
```

### Input
Assessment objective: {objective}
{concept_num} most critical concepts about {objective}:
"""


INS_TEMP = """You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' understanding of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the {level} level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": {level}, "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. Ensure that the correct answer and supporting evidence are available within the provided document. Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. The options should seem like reasonable answers to the question, but are actually incorrect. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
Only respond with the questions in json format without any other content.
"""


MUL_REMEMBER_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of remembering. This is the most basic level. It involves recalling or recognizing facts, terms, basic concepts, or answers without necessarily understanding what they mean. Examples include memorizing a formula, a definition, or a quote. For example, what is the capital city of France?


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the remembering level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "remembering", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required remembering level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example1
Topic: Definition of homeostasis

<Start of Provided Document>
In biology, homeostasis (British also homoeostasis) is the state of steady internal, physical, chemical, and social conditions maintained by living systems. This is the condition of optimal functioning for the organism and includes many variables, such as body temperature and fluid balance, being kept within certain pre-set limits (homeostatic range).
<End of Provided Document>

Question of remembering level in Bloom's Taxonomy:
```json
[{{"level": "remembering", "question": "What is homeostasis in biology?", "A": "The process by which living systems acquire energy from their environment", "B": "The state of steady internal conditions maintained by living systems", "C": "The ability of living systems to adapt to external environmental changes", "D": "The mechanism through which living organisms regulate their cell number", "answer": "B", "explanation": "As stated in the provided document, Homeostasis is defined as the state of steady internal, physical, chemical, and social conditions maintained by living systems, "}}]
```


### Example2
Topic: Apple

<Start of Provided Document>
An apple is a round, edible fruit produced by an apple tree (Malus spp., among them the domestic or orchard apple; Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Asia and Europe and were introduced to North America by European colonists.
<End of Provided Document>

Question of remembering level in Bloom's Taxonomy:
```json
[{{"level": "remembering", "question": "Where did the apple tree originate?", "A": "East Asia", "B": "West Asia", "C": "Central Asia", "D": "South America", "answer": "C", "explanation": "The apple tree originated in Central Asia, as mentioned in the provided document."}}]
```


### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of remembering level in Bloom's Taxonomy:
"""


MUL_UNDERSTAND_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of understanding. It involves demonstrating an understanding of facts and ideas by organizing, comparing, translating, interpreting, giving descriptions, and stating the main ideas. For example, explain why the French Revolution started?


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the understanding level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "understanding", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required understanding level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example1
Topic: Lion

<Start of Provided Document>
The lion (Panthera leo) is a large cat of the genus Panthera native to Africa and India. It has a muscular, broad-chested body; short, rounded head; round ears; and a hairy tuft at the end of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane.
<End of Provided Document>

Question of understanding level in Bloom's Taxonomy:
```json
{{"level": "understanding", "question": "What does it mean that the lion is 'sexually dimorphic'?", "A": "Male and female lions have different mating behaviors.", "B": "Male and female lions exhibit different hunting techniques.", "C": "Male and female lions live in separate social groups.", "D": "Male and female lions have different physical characteristics.", "answer": "D", "explanation": "Accoring to the document, sexually dimorphic means that adult male lions are larger than females and have a prominent mane, indicating different physical characteristics between the sexes."
}}
```


### Example2
Topic: Apple

<Start of Provided Document>
You may easily recognize Granny Smith apples by their bright green color and slightly tart flavor. Research shows that they contain no anthocyanidins, as demonstrated by their lack of red, blue, or purple color. Still, they have other benefits to offer (1Trusted Source). One animal study found that fiber from Granny Smith apples could modify gut microbiota profiles in mice with obesity to resemble those of lean mice, suggesting a potential weight control capacity (9Trusted Source).
<End of Provided Document>

Question of understanding level in Bloom's Taxonomy:
```json
{{"level": "understanding", "question": "Which of the following varieties of apples is characterized by bright green color and slightly tart flavor?", "A": "Granny Smith apples", "B": "Red Delicious apples", "C": "Golden Delicious apples", "D": "Fuji apples", "answer": "A", "explanation": "As mentioned in the provided document, you can recognize Granny Smith apples by their bright green color and slightly tart flavor."}}
```



### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of understanding level in Bloom's Taxonomy:\n
"""





MUL_APPLY_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of applying. This level involves using acquired knowledge—solving problems in new situations by applying acquired knowledge, facts, techniques, and rules in a different, or new way. For example, would apples prevent scurvy, a disease caused by a deficiency in vitamin C?


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the applying level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "applying", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required applying level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example1
Topic: Lion

<Start of Provided Document>
The lion inhabits grasslands, savannahs and shrublands. It is usually more diurnal than other wild cats, but when persecuted, it adapts to being active at night and at twilight.
<End of Provided Document>

Question of applying level in Bloom's Taxonomy:
```json
[{{"level": "applying", "question": "If a lion population is located in a region with increased human activity, what behavioral adaptation might they exhibit?", "A": "Become more nocturnal", "B": "Hunt in larger groups", "C": "Reduced territory range", "D": "More frequent relocation", "answer": "A",
"explanation": "Accoring to the document, when persecuted, the lion adapts to being active at night and at twilight."}}]
```


### Example2
Topic: Apple

<Start of Provided Document>
Apples are high in fiber, vitamin C, and various antioxidants. They are also very filling, considering their low calorie count. StudiesTrusted Source showTrusted Source that eating apples can have multiple benefits for your health.
<End of Provided Document>

Question of applying level in Bloom's Taxonomy:
```json
[{{"level": "applying", "question": "Would apples prevent scurvy, a disease caused by a deficiency in vitamin C?", "A": "No, apples are not drugs.", "B": "Yes, apples contain vitamin C", "C": "No, apples do not contain vitamin C", "D": "Not sure", "answer": "B", "explanation": "As mentioned in the provided document, apples are high in fiber, vitamin C, and various antioxidants."}}]
```


### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of applying level in Bloom's Taxonomy:
"""


MUL_ANALYZ_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of analyzing. Analyzing involves examining and breaking information into component parts, determining how the parts relate to one another, identifying motives or causes, making inferences, and finding evidence to support generalizations. For example, compare and contrast four ways of serving foods made with apples and examine which ones have the highest health benefits.


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the analyzing level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "analyzing", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required analyzing level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example1
Topic: Lion

<Start of Provided Document>
The lion inhabits grasslands, savannahs and shrublands. It is usually more diurnal than other wild cats, but when persecuted, it adapts to being active at night and at twilight.
<End of Provided Document>

Question of analyzing level in Bloom's Taxonomy:
```json
[{{"level": "analyzing", "question": "Which factor has primarily contributed to the decline in lion populations?", "A": "Natural climate changes", "B": "Habitat loss and conflicts with humans", "C": "Decreased prey availability", "D": "Intrinsic genetic problems", "answer": "B","explanation": "As mentioned in the provided document, the primary factors contributing to the decline in lion populations are habitat loss and conflicts with humans."}}]
```


### Example2
Topic: Apple

<Start of Provided Document>
In summary, the health benefits of apple-based foods largely depend on how they are prepared and what additional ingredients are added. Raw or minimally processed forms retain the most nutrients and are the healthiest options.
<End of Provided Document>

Question of analyzing level in Bloom's Taxonomy:
```json
[{{"level": "analyzing", "question": "Which apple-based food have the highest health benefits?", "A": "Apple Pie", "B": "Baked Apples", "C": "Apple Juice", "D": "Raw Apple", "answer": "D", "explanation": "As mentioned in the provided document, raw or minimally processed forms retain the most nutrients and are the healthiest options."}}]
```


### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of analyzing level in Bloom's Taxonomy:\n
"""



MUL_EVAL_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of evaluating. This involves presenting and defending opinions by making judgments about information, the validity of ideas, or quality of work based on a set of criteria. For example, which kinds of apples are suitable for baking a pie, and why?


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the evaluating level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "evaluating", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required evaluating level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example
Topic: Apple

<Start of Provided Document>
Granny Smith apples are tart and firm. They hold their shape well during baking and provide a nice contrast to the sweet filling. Red Delicious apples tend to become mushy and mealy when baked. They lack the necessary tartness and firmness for a good pie.
<End of Provided Document>

Question of evaluating level in Bloom's Taxonomy:
```json
[{{"level": "evaluating", "question": "Which kinds of apples are suitable for baking a pie", "A": "Red Delicious", "B": "McIntosh", "C": "Granny Smith", "D": "Empire", "answer": "D", "explanation": "As mentioned in the provided document, Granny Smith apples are tart and firm. They hold their shape well during baking and provide a nice contrast to the sweet filling."}}]
```


### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of evaluating level in Bloom's Taxonomy:\n
"""



MUL_CREATE_INS = """### Background
According to Bloom's Taxonomy, the educational objectives are categorized into six hierarchical levels including remembering, understanding, applying, analyzing, evaluating and creating. Now you must focus on the level of creating. Creating involves putting elements together to form a coherent or functional whole; reorganizing elements into a new pattern or structure through generating, planning, or producing.  For example, design a research study to test the impact of social media on academic performance.


### Instruction
You are now an educator and hope to use Bloom's Taxonomy to assess a group of very knowledgeable students' knowledge of specific knowledge points. Specifically, for a given topic and document, you need to generate 5 difficult questions about the topic corresponding the creating level in Bloom's Taxonomy. 
The reponse must follow the provided json format: [{{"level": "creating", "question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]
Ensure the question can be answered independently without additional context. 
Ensure that the correct answer and supporting evidence are available within the provided document. 
The explanation must begin with "According to the document,".
Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
Provide four comprehensive options for each question, with only one being the correct choice. 
You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
You will be penalized for asking questions that contain multiple different answers.
You will be penalized for proposing questions that don't meet the required creating level in Bloom's Taxonomy.
Only respond with the questions in json format without any other content.


### Example 1
Topic: Apple Pie

<Start of Provided Document>
At high altitudes, bakers face unique challenges due to the reduced air pressure. In these conditions, leavening gases in baked goods, like those produced by baking powder, expand more than they would at sea level. This increased expansion can cause the crusts of breads and pastries to rise excessively, leading to an overly airy or dry texture. To counteract this, it's advisable to reduce the amount of baking powder used in recipes when baking at high altitudes. This adjustment helps maintain the desired texture and prevents the crust from becoming too light and brittle.
<End of Provided Document>

Question of creating level in Bloom's Taxonomy:
```json
[{{"level": "creating", "question": "How would you adjust a traditional apple pie recipe to make it suitable for a high-altitude baking environment?", "A": "Increase the amount of sugar in the filling.", "B": "Use more apples to increase moisture.",
"C": "Decrease the baking temperature by 25°F.","D": "Decrease the amount of baking powder used in the crust.", "answer": "D",
"explanation": "According to the provided document, at high altitudes, leavening gases expand more, so reducing baking powder prevents the crust from rising too much and becoming too airy or dry."}}]
```

### Example 2
Topic: Apple Pie

<Start of Provided Document>
Finely chopped nuts, such as almonds, walnuts, or pecans, can significantly enhance the sensory experience of a pie. When incorporated into a pie, these nuts provide a subtle yet distinctive crunch, which contrasts beautifully with the typically soft and flaky nature of the pie crust. This textural variation adds an element of surprise and delight to each bite, elevating the overall eating experience. 
<End of Provided Document>

Question of creating level in Bloom's Taxonomy:
```json
[{{"level": "creating", "question": "How would you adjust a traditional apple pie recipe to make it suitable for a high-altitude baking environment?", "A": "Add extra butter to the dough", "B": "Blend oats into the crust mixture",
"C": "Increase the amount of water in the dough","D": "Incorporate finely chopped nuts into the crust dough", "answer": "D",
"explanation": "According to the provided document, finely chopped nuts would provide a subtle crunch, offering a textural contrast to the typically soft and flaky pie crust."}}]
```


### User Input
Topic: {topic}

<Start of Provided Document>
{document}
<End of Provided Document>

5 difficult questions with one correct answer and three misleading options of creating level in Bloom's Taxonomy:\n
"""


CONCEPT_GENRATION_PROMPT = """### Instruction
You are now an excellent educator and hope to assess a group very knowledgeable students' knowledge. Specifically, you are given a seed question, a critical concept related to the question, and a corresponding document. Then you need to generate 3 difficult questions about the concept, and these questions should help evaluate whether the test-taker truly possesses knowledge related to the seed question.
The reponse must follow the provided json format: [{{"question": <question>, "A": <choice_A>, "B": <choice_B>, "C": <choice_C>, "D": <choice_D>, "answer": <answer>, "explanation": <explanation>}}, ...]


### Rules
1. Ensure the question can be answered independently without additional context.
2. The correct answer and supporting evidence are available must appear in the provided document.
3. Ask question that are exclusively single-choice and have a clear, correct answer which can be find the the provided document. 
4. Provide 4 comprehensive options for each question, with only 1 being the correct choice. 
5. You will be penalized for proposing too easy question. Every question must be fully understood with the corresponding knowledge in order to be answered correctly, and it cannot be guessed. In order to increase the difficulty of answering the question correctly, you must design three incorrect options with strong misleadingness. The three incorrect options must be similar with the correct answer. You must provide options that you would confuse with the correct answer yourself.
6. You will be penalized for including statements like "in the document" or "according to the provided document" or "as mentioned in the document" in the question.
7. You will be penalized for asking questions that contain multiple different answers.
8. Only respond with the questions in json format without any other content.
9. The explanation must begin with "According to the document,"


### Example1
Concept: Red Delicious

<Start of Seed Question>
Question: Which of the following varieties of apples is characterized by bright green color and slightly tart flavor?
A. Granny Smith
B. Red Delicious
C. Golden Delicious
D. Fuji
Answer: A
<End of Seed Question>

<Start of Provided Document>
Red Delicious is a type of apple with a red exterior and sweet taste that was first recognized in Madison County, Iowa, in 1872. Today, the name Red Delicious comprises more than 50 cultivars. It was the most produced cultivar in the United States from 1968 to 2018, when it was surpassed by Gala.
<End of Provided Document>


Question about the concept "Red Delicious":
```json
{{"question": "What is the color and flavor of Red Delicious apples?", "A": "a red exterior and sweet taste", "B": "bright green color and slightly tart flavor", "C": "golden color and sweet taste", "D": "red color and sightly tart flavor", "answer": "A", "explanation": "According to the document, Red Delicious is a type of apple with a red exterior and sweet taste."}}
```

### Example2
Concept: Solar energy

<Start of Seed Question>
Question: Which of the following countries generated the most total energy from solar sources in 2019?
A. China
B. United States
C. Germany
D. Japan
Answer: A. China
<End of Seed Question>

<Start of Provided Document>
The total solar energy absorbed by Earth's atmosphere, oceans and land masses is approximately 122 PW·year = 3,850,000 exajoules (EJ) per year. In 2002 (2019), this was more energy in one hour (one hour and 25 minutes) than the world used in one year. 
<End of Provided Document>

Question about the concept "Solar energy":
```json
{{"question": "What is the approximate total solar energy absorbed by Earth's atmosphere, oceans, and land masses per year?", "A": "1,220 PW·year", "B": "3,850,000 exajoules", "C": "3,000 exajoules", "D": "450,000 km2", "answer": "B", "explanation": "According to the document, the total solar energy absorbed by Earth's atmosphere, oceans and land masses is approximately 122 PW·year = 3,850,000 exajoules (EJ) per year."}}
```


### User Input
Concept: {concept}

<Start of Seed Question>
{question}
<End of Seed Question>

<Start of Provided Document>
{document}
<End of Provided Document>

3 difficult questions with 1 correct answer and 3 misleading options about the concept "{concept}":\n
"""

