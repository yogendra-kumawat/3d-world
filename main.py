from tactile_vision.ui.app import build_app


def main() -> None:
    app = build_app()
    app.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
