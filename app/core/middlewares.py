from app.core.pipes.VLLMPipe import VLLMPipe


def contextualize(msg):
    def prompt_appender(args):
        current_pipe = args['current_pipe']
        if current_pipe.__class__.__name__ == VLLMPipe.__name__:
            return (f"you are a helpful airport assistant for passengers, "
                    f"given the following context: {args['last_result']}, "
                    f"answer the following question of the passenger: {msg}")
        return args['last_result']

    return prompt_appender
