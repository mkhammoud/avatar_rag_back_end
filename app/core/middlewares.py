from app.core.pipes.VLLMPipe import VLLMPipe


def contextualize(msg):
    def prompt_appender(args):
        current_pipe = args['current_pipe']
        if current_pipe.__class__.__name__ == VLLMPipe.__name__:
            return f"context: {args['last_result']}, answer the following: {msg}"
        return args['last_result']

    return prompt_appender
