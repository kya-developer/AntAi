#main.py
from predict import Predict
from pyfiglet import Figlet

def ascii_art():
    f = Figlet()
    banner = (f.renderText('Ant Ai'))
    print(banner)



def main():
    ascii_art()
    while True:
        command = input(">>> ")

        if command == "exit":
            break

        process_command(command)


def process_command(command):

    if command == "help":
        print("Commands:")
        print("help - Show this help message")
        print("exit - Exit the program")
        print("predict <tempurature> <humidity> <light> <time> <species> - Predict the behavior of an ant based on the given parameters")
        return
    elif command == "predict":
        print("please answer the following questions:")
        tempurature = float(input("tempurature: "))
        humidity = float(input("humidity: "))
        light = input("light (True/False): ").lower() == "true"
        time = int(input("time (0-23): "))
        species = input("species: ")

        label = Predict(tempurature, humidity, light, time, species)
        print(f"Predicted behavior: {label}")
    else:
        print("Unknown command. Type 'help' for a list of commands.")


if __name__ == "__main__":
    main()