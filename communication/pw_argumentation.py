from mesa import Model
from mesa.time import RandomActivation
from communication.agent.CommunicatingAgent import CommunicatingAgent
from communication.message.MessageService import MessageService

class ArgumentAgent ( CommunicatingAgent ) :
    """ ArgumentAgent which inherit from CommunicatingAgent .
    """
    def __init__ ( self, unique_id, model, name, preferences ) :
        super ().__init__ ( unique_id, model, name, preferences )
        self.preference = preferences

    def step ( self ) :
        super ().step ()

    def get_preference ( self ) :
        return self.preference

    def generate_preferences ( self, List_items ) :
        # see question 3
        # To be completed
        pass

class ArgumentModel ( Model ) :
    """ ArgumentModel which inherit from Model .
    """
    def __init__(self) :
        self.schedule = RandomActivation(self)
        self.__messages_service = MessageService(self.schedule)

        # To be completed
        #
        # a = ArgumentAgent ( id, " agent_name ")
        # a.generate_preferences ( preferences )
        # self.schedule.add ( a )
        # ...

        self.running = True

    def step ( self ) :
        self.__messages_service.dispatch_messages ()
        self.schedule.step ()


if __name__ == " __main__ " :
    argument_model = ArgumentModel ()

    # To be completed