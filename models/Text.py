class_names_5 = [
'Neutrality in learning state.',
'Enjoyment in learning state.',
'Confusion in learning state.',
'Fatigue in learning state.',
'Distraction.'
]

class_names_with_context_5 = [
'an expression of Neutrality in learning state.',
'an expression of Enjoyment in learning state.',
'an expression of Confusion in learning state.',
'an expression of Fatigue in learning state.',
'an expression of Distraction.'
]


##### onlyface
class_descriptor_5_only_face = [
'Relaxed mouth,open eyes,neutral eyebrows,smooth forehead,natural head position.',

'Upturned mouth,sparkling or slightly squinted eyes,raised eyebrows,relaxed forehead.',

'Furrowed eyebrows, slightly open mouth, squinting or narrowed eyes, tensed forehead.',

'Mouth opens in a yawn, eyelids droop, head tilts forward.',

'Averted gaze or looking away, restless or fidgety posture, shoulders shift restlessly.'
]

##### with_context
class_descriptor_5 = [
'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',

'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',

'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',

'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',

'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
]

# ================= DAiSEE EMOTIONAL MAPPING =================
# Mapping Engagement Levels (0-3) to Emotional States

class_names_daisee = [
    'Boredom and Distraction',      # Level 0
    'Fatigue and Passivity',        # Level 1
    'Calm Attention',               # Level 2
    'Strong Interest and Curiosity' # Level 3
]

class_names_with_context_daisee = [
    'a student feeling bored, distracted, or looking away from the screen.',
    'a student feeling tired, passive, sleepy, or zoning out.',
    'a student paying attention, looking calm and focused on the screen.',
    'a student showing strong interest, curiosity, and active engagement.'
]

class_descriptor_daisee = [
    'Face showing boredom, yawning, eyes looking away, head turning around, completely disengaged.',
    'Face showing fatigue, sleepy eyes, blank stare, resting head on hand, passive expression.',
    'Face showing calmness, serious expression, direct eye contact with screen, normal posture.',
    'Face showing excitement, raising eyebrows, leaning forward, intense focus, nodding or smiling.'
]