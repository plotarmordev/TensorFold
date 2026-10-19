# Run the server; when it stops on SIGINT, print every thread's stack, then batch mode ends and kills it.
set pagination off
handle SIGPIPE nostop noprint pass
run
thread apply all bt
