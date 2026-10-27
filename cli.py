"""Chat with the agent in your terminal:  python cli.py"""
from agent.agent import SupportAgent


def main():
    agent = SupportAgent()
    print("Koala Gear support (type 'quit' to exit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        r = agent.handle(msg)
        print(f"Bot: {r.reply}")
        print(f"     [intent={r.intent} conf={r.confidence} action={r.action}"
              f"{' ESCALATED' if r.escalated else ''}{' sources=' + ','.join(r.sources) if r.sources else ''}]\n")


if __name__ == "__main__":
    main()
