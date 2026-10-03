import sys
import subprocess


def main():
    print("🧠 Memulai DocuMind Web Interface...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDocuMind dimatikan. Sampai jumpa!")


if __name__ == "__main__":
    main()
