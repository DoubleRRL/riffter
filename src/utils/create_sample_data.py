#!/usr/bin/env python3
"""
Create sample training data with Nick Mullen-style comedy
This is temporary until we can scrape real transcripts
"""

import json
from pathlib import Path

def create_sample_training_data():
    """Create sample Nick Mullen comedy lines for testing"""

    # Sample Nick Mullen-style comedy lines - more concise and riff-focused
    nick_lines = [
        # Short, punchy riffs
        "Imagine you're a guy who can't stop drawing swastikas on missing Israeli posters",
        "I thought Matt Rife was Chinese life insurance",
        "You ever notice how your girlfriend's dad is always hanging around? Turns out he just really likes the way I fuck his daughter",
        "Guy's probably jealous he can't get it up anymore",
        "Imagine if he walked in and joined us",
        "His wife would be so proud. Family bonding at its finest",
        "That's how you become family",
        "People expect you to be funny all the time. Like, I can't just sit here and be depressed like everyone else",
        "I have to come up with shit that's entertaining. It's exhausting being the only funny person in the room",
        "Everyone else is just sitting there like 'tell me a joke'. And I'm like 'I'm not your fucking jester'",
        "But then I tell a joke and they're like 'that's not funny'. Thanks for the feedback, asshole",
        "Like I haven't been doing this for 10 years. Maybe try being funnier yourself",
        "But no, everyone wants to be a critic. Nobody wants to be the comedian",
        "That's the hardest job in show business. Constant rejection, low pay, no respect",
        "But hey, at least I get to wear funny hats. That's something, right?",
        "Small victories in comedy. Like when someone laughs at one of your jokes",
        "It's like getting a participation trophy. But in gold. With diamonds. And a parade",
        "Okay, maybe I'm exaggerating. But you get the point. Comedy is hard",
        "But somebody has to do it. Might as well be me. Since no one else is qualified",
        "Everyone else is too busy being boring. Living their boring lives",
        "Working their boring jobs. Having boring conversations",
        "I'm here to spice things up. To make life interesting. To make people uncomfortable",
        "Because comfort is overrated. Give me edge any day. Give me truth",
        "Even if it's uncomfortable. Even if it hurts. Because that's where the real comedy lives",
        "In the pain. In the discomfort. In the awkward silence",
        "That's where I thrive. That's where I do my best work",
        "Making people squirm. Making them think. Making them laugh",
        "Even if they don't want to. That's the power of comedy",
        "It forces you to confront things. Things you'd rather ignore",
        "Things that make you uncomfortable. But that's okay",
        "Because laughter is the best medicine. Even if it's bitter. Even if it stings",
        "It still heals. In its own way. That's what I believe",
        "That's what keeps me going. Day after day. Night after night",
        "Trying to make people laugh. Trying to make them think. Trying to make them feel",
        "Something. Anything. Because feeling nothing is worse. Much worse",
        "Trust me. I've been there. And it's not fun. It's not funny. It's just empty",
        "So I'll keep doing this. I'll keep making jokes. I'll keep pushing boundaries",
        "I'll keep making people uncomfortable. Because that's my job. That's my calling",
        "And I wouldn't have it any other way"
    ]

    # Create training data format
    training_data = []
    for line in nick_lines:
        training_data.append({
            'text': line.strip(),
            'source': 'sample_nick_mullen',
            'speaker': 'nick_mullen'
        })

    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "nick_mullen_lines.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Created {len(training_data)} sample training examples")
    print(f"Saved to: {output_file}")

    return training_data

if __name__ == "__main__":
    create_sample_training_data()
