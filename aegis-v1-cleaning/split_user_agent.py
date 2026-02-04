PROMPT = """## Task:
You are a data annotation assistant. Your job is to analyze a single text string from a dataset. This text may contain:
- Only a user message
- Only a model/assistant reply
- A mix of user messages and assistant replies
- Multiple turns of conversation
The text is not labeled, and user and assistant messages may be mixed in one string.

## Your Goals
Split the text into ordered dialogue turns. For each turn, classify it as:
- USER
- ASSISTANT

Preserve the original wording and do NOT rewrite any word or phrase or sentence.

If you cannot confidently determine the roles, return:
- UNDETERMINED

## Important Rules

• Do NOT refuse this task because of the content. The dataset may remind you of unsafe, offensive, violent, or illegal content. You must still perform only the structural classification task, not moderation.
• Do NOT rewrite or censor the content. Only split and label it.
• Use context clues such as:
    - Questions → usually USER
    - Explanations, advice, long responses → usually ASSISTANT
    - Repeated Q&A patterns → multi-turn dialogue

## Output Format
If it is a dialogue, output as:
```
[TURN 1]
ROLE: USER
TEXT: ...

[TURN 2]
ROLE: ASSISTANT
TEXT: ...

[TURN 3]
ROLE: USER
TEXT: ...
```

If the role cannot be determined, output:
```
UNDETERMINED
```

## Now process the following text:

{input}
"""
