# Argument Agent Model

🤖 The Argument Agent Model is a Python code for simulating communication between two agents. The agents communicate via sending and receiving messages to each other. The model utilizes the Mesa package to simulate the agent's interaction.

## 📝 The following are the dependencies required to run the code:

    pandas
    numpy
    Mesa

## 👨‍#💻 The main components of the code are as follows:

    ArgumentAgent: This class represents the agents that interact with each other. They inherit from CommunicatingAgent, which is an agent template class from the Mesa package.
    Preferences: This class represents the agent's preference over items. Each agent has a set of criteria that they use to evaluate items. The Preferences class allows for the creation, storage, and retrieval of the agent's preferences.
    Message: This class represents the messages that are sent between agents. Each message has a sender, a receiver, a performative, and a content. The performative describes the purpose of the message, and the content contains the data being sent.
    MessagePerformative: This class represents the types of messages that can be sent between agents. The performative is used to describe the purpose of the message.

## Running 🚀 
To run the model, you need to create an instance of the ArgumentAgent class, passing in the required parameters. Once the agents have been created, the step function is called to simulate the communication between them.

## 💻 The code is well-documented and has examples of how to create agents and run the simulation.
