"""Interface Tkinter para conversar com o Ollama e falar as respostas."""

from __future__ import annotations

import base64
import importlib
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from core import DEFAULT_MODEL, VISION_MODEL, append_conversation, ask_local_ai, try_local_tool
from image_generation import extract_image_prompt
from pexels_images import search_pexels_image


class AssistantApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Assistente local")
        self.root.geometry("760x560")
        self.root.minsize(560, 400)

        self.history: list[dict[str, str]] = []
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.conversation_path = Path(__file__).resolve().parent / "conversas" / "conversa.txt"
        self._last_user_message = ""
        self.attached_photos: list[tuple[str, bytes]] = []
        self.transcript_images: list[tk.PhotoImage] = []

        self.model = tk.StringVar(value=DEFAULT_MODEL)
        self.speak_responses = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Pronto")

        self._build_interface()
        self.root.after(100, self._process_events)

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        settings = ttk.Frame(self.root, padding=(12, 12, 12, 8))
        settings.grid(row=0, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Modelo Ollama:").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(settings, textvariable=self.model).grid(row=0, column=1, sticky="ew")
        ttk.Checkbutton(
            settings,
            text="Falar respostas",
            variable=self.speak_responses,
        ).grid(row=0, column=2, padx=(12, 0))

        self.transcript = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=10,
            pady=10,
        )
        self.transcript.grid(row=1, column=0, sticky="nsew", padx=12)

        composer = ttk.Frame(self.root, padding=12)
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(2, weight=1)

        ttk.Button(composer, text="Anexar foto", command=self._choose_photos).grid(
            row=0, column=0, padx=(0, 6), pady=(0, 8)
        )
        ttk.Button(composer, text="Tirar foto", command=self._open_camera).grid(
            row=0, column=1, padx=(0, 8), pady=(0, 8)
        )
        self.photo_summary = ttk.Label(composer, text="Nenhuma foto anexada")
        self.photo_summary.grid(row=0, column=2, sticky="ew", pady=(0, 8))
        ttk.Button(composer, text="Limpar fotos", command=self._clear_photos).grid(
            row=0, column=3, padx=(8, 0), pady=(0, 8)
        )

        self.message = ttk.Entry(composer)
        self.message.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 8))
        self.message.bind("<Return>", self._send_message)

        self.send_button = ttk.Button(composer, text="Enviar", command=self._send_message)
        self.send_button.grid(row=1, column=3)

        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=(12, 0, 12, 10)).grid(
            row=3, column=0, sticky="ew"
        )

    def _append_text(self, speaker: str, message: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"{speaker}: {message}\n\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _append_image(self, photo: tk.PhotoImage, caption: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"{caption}\n")
        self.transcript.image_create(tk.END, image=photo, align=tk.TOP)
        self.transcript.insert(tk.END, "\n\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)
        self.transcript_images.append(photo)
        self.transcript_images = self.transcript_images[-12:]

    def _photo_preview(self, image_data: bytes) -> tk.PhotoImage:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
        frame = cv2.imdecode(numpy.frombuffer(image_data, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não consegui abrir esta imagem. Use JPG, PNG ou WEBP.")

        height, width = frame.shape[:2]
        scale = min(1.0, 480 / width, 320 / height)
        if scale < 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise ValueError("Não consegui preparar a prévia desta imagem.")
        image_string = base64.b64encode(encoded.tobytes()).decode("ascii")
        return tk.PhotoImage(data=image_string, format="png")

    def _update_photo_summary(self) -> None:
        count = len(self.attached_photos)
        if count == 0:
            summary = "Nenhuma foto anexada"
        else:
            names = ", ".join(name for name, _ in self.attached_photos[:2])
            summary = f"{count} foto(s): {names}"
            if count > 2:
                summary += ", ..."
        self.photo_summary.configure(text=summary)

    def _choose_photos(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Escolha fotos do jardim",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        for path in paths:
            if len(self.attached_photos) >= 4:
                self._append_text("Aviso", "Você pode enviar até quatro fotos por vez.")
                break
            try:
                data = Path(path).read_bytes()
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError("Cada foto precisa ter menos de 20 MB.")
                self.attached_photos.append((Path(path).name, data))
            except (OSError, ValueError) as error:
                self._append_text("Aviso", str(error))
        self._update_photo_summary()

    def _clear_photos(self) -> None:
        self.attached_photos.clear()
        self._update_photo_summary()

    def _open_camera(self) -> None:
        try:
            cv2 = importlib.import_module("cv2")
            camera = cv2.VideoCapture(0)
        except (ImportError, OSError) as error:
            self._append_text("Erro de câmera", str(error))
            return

        if not camera.isOpened():
            camera.release()
            self._append_text("Erro de câmera", "Não encontrei uma câmera disponível.")
            return

        window = tk.Toplevel(self.root)
        window.title("Tirar foto do jardim")
        preview = ttk.Label(window, text="Abrindo câmera...")
        preview.pack(padx=12, pady=12)
        controls = ttk.Frame(window, padding=(12, 0, 12, 12))
        controls.pack(fill=tk.X)
        frame_state: dict[str, object | None] = {"frame": None, "photo": None}
        closed = False

        def close_camera() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            camera.release()
            window.destroy()

        def refresh_preview() -> None:
            if closed or not window.winfo_exists():
                return
            success, frame = camera.read()
            if success:
                frame_state["frame"] = frame
                height, width = frame.shape[:2]
                scale = min(1.0, 640 / width, 480 / height)
                if scale < 1:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                png_success, encoded = cv2.imencode(".png", frame)
                if png_success:
                    photo = tk.PhotoImage(
                        data=base64.b64encode(encoded.tobytes()).decode("ascii"),
                        format="png",
                    )
                    preview.configure(image=photo, text="")
                    preview.image = photo
                    frame_state["photo"] = photo
            window.after(40, refresh_preview)

        def capture_photo() -> None:
            frame = frame_state["frame"]
            if frame is None:
                self._append_text("Câmera", "Ainda não há uma imagem da câmera.")
                return
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
            if not success:
                self._append_text("Câmera", "Não consegui salvar a captura em memória.")
                return
            self._add_photo("câmera", encoded.tobytes())
            close_camera()

        ttk.Button(controls, text="Capturar", command=capture_photo).pack(side=tk.LEFT)
        ttk.Button(controls, text="Cancelar", command=close_camera).pack(side=tk.RIGHT)
        window.protocol("WM_DELETE_WINDOW", close_camera)
        refresh_preview()

    def _add_photo(self, name: str, data: bytes) -> None:
        if len(self.attached_photos) >= 4:
            self._append_text("Aviso", "Você pode enviar até quatro fotos por vez.")
            return
        self.attached_photos.append((name, data))
        self._update_photo_summary()

    def _encode_photos(self, photos: list[tuple[str, bytes]]) -> list[str]:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
        encoded_photos: list[str] = []

        for name, image_data in photos:
            image = cv2.imdecode(numpy.frombuffer(image_data, dtype=numpy.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Não consegui abrir a foto {name}. Use JPG, PNG ou WEBP.")

            height, width = image.shape[:2]
            scale = min(1.0, 1536 / max(width, height))
            if scale < 1:
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
            success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not success:
                raise ValueError(f"Não consegui preparar a foto {name} para análise.")
            encoded_photos.append(base64.b64encode(encoded.tobytes()).decode("ascii"))

        return encoded_photos

    def _send_message(self, _event: tk.Event | None = None) -> None:
        message = self.message.get().strip()
        photos = self.attached_photos.copy()
        if (not message and not photos) or str(self.send_button["state"]) == tk.DISABLED:
            return
        if photos and not message:
            message = "Analise estas fotos do meu jardim e sugira melhorias práticas."

        self.message.delete(0, tk.END)
        self._last_user_message = message
        photo_suffix = f" ({len(photos)} foto(s))" if photos else ""
        self._append_text("Você", f"{message}{photo_suffix}")
        for name, image_data in photos:
            try:
                self._append_image(self._photo_preview(image_data), f"Foto: {name}")
            except (ImportError, OSError, ValueError) as error:
                self._append_text("Prévia indisponível", str(error))
        self.attached_photos.clear()
        self._update_photo_summary()
        append_conversation(self.conversation_path, "user", message)

        history = self.history.copy()
        model = self.model.get().strip() or DEFAULT_MODEL
        speak = self.speak_responses.get()
        self.send_button.configure(state=tk.DISABLED)
        self.status.set("Consultando o assistente...")

        threading.Thread(
            target=self._answer_message,
            args=(message, history, model, speak, photos),
            daemon=True,
        ).start()

    def _answer_message(
        self,
        message: str,
        history: list[dict[str, str]],
        model: str,
        speak: bool,
        photos: list[tuple[str, bytes]],
    ) -> None:
        try:
            image_data = self._encode_photos(photos) if photos else None
            image_prompt = extract_image_prompt(message)

            if image_prompt is not None:
                if image_data:
                    self.events.put(("status", "Analisando o jardim para criar o conceito..."))
                    garden_description = ask_local_ai(
                        "Descreva o layout, as plantas e os elementos visíveis neste jardim para orientar uma proposta visual.",
                        history,
                        VISION_MODEL,
                        images=image_data,
                    )
                    image_prompt = (
                        f"Garden design inspired by this real garden description: {garden_description}. "
                        f"Requested concept: {image_prompt}"
                    )

                self.events.put(("status", "Buscando uma foto real no Pexels..."))
                photo = search_pexels_image(
                    image_prompt,
                    Path(__file__).resolve().parent / "imagens_pexels",
                )
                relative_path = photo.path.relative_to(Path(__file__).resolve().parent)
                answer = (
                    f"Encontrei uma foto real: {photo.alt}. "
                    f"Fotógrafo: {photo.photographer}. "
                    f"Fonte: {photo.page_url}. "
                    f"Arquivo salvo em {relative_path}."
                )
                if photos:
                    answer += " A busca foi inspirada na descrição da sua foto; não é uma edição dela."
                self.events.put(("pexels_image", str(photo.path)))
            elif image_data:
                self.events.put(("status", "Analisando as fotos com o modelo visual..."))
                answer = ask_local_ai(message, history, VISION_MODEL, images=image_data)
            else:
                answer = try_local_tool(message)
                if answer is None:
                    answer = ask_local_ai(message, history, model)

            append_conversation(self.conversation_path, "assistant", answer)
            self.events.put(("answer", answer))

            if speak:
                self.events.put(("status", "Falando a resposta..."))
                try:
                    speech_module = importlib.import_module("pyttsx3")
                    engine = getattr(speech_module, "init")()
                    engine.say(answer)
                    engine.runAndWait()
                except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                    self.events.put(("speech_error", str(error)))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.events.put(("error", str(error)))
        finally:
            self.events.put(("done", ""))

    def _process_events(self) -> None:
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "answer":
                self._append_text("Assistente", value)
                self.history.extend(
                    [
                        {"role": "user", "content": self._last_user_message},
                        {"role": "assistant", "content": value},
                    ]
                )
                self.history = self.history[-12:]
            elif event == "error":
                self._append_text("Erro", value)
            elif event == "speech_error":
                self._append_text("Aviso de voz", value)
            elif event == "generated_image":
                try:
                    image = tk.PhotoImage(file=value)
                    scale = max(1, (image.width() + 479) // 480)
                    if scale > 1:
                        image = image.subsample(scale, scale)
                    self._append_image(image, "Conceito gerado")
                except tk.TclError as error:
                    self._append_text("Imagem gerada", f"Arquivo salvo, mas não consegui mostrar a prévia: {error}")
            elif event == "pexels_image":
                try:
                    image = self._photo_preview(Path(value).read_bytes())
                    self._append_image(image, "Foto real do Pexels")
                except (ImportError, OSError, ValueError) as error:
                    self._append_text(
                        "Foto do Pexels",
                        f"A foto foi salva, mas não consegui mostrar a prévia: {error}",
                    )
            elif event == "status":
                self.status.set(value)
            elif event == "done":
                self.send_button.configure(state=tk.NORMAL)
                self.status.set("Pronto")

        self.root.after(100, self._process_events)


if __name__ == "__main__":
    app_root = tk.Tk()
    app = AssistantApp(app_root)
    app_root.mainloop()