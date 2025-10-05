import json
import logging
import os
import asyncio
from datetime import datetime
import threading
import time
import hashlib
import colorsys
from typing import Optional
import tkinter as tk
from tkinter import scrolledtext, simpledialog, filedialog, messagebox
from tkinter import ttk

import discord
import requests

from config_loader import (
    load_globals,
    load_pdvs,
    load_classes,
    load_agents,
    save_globals,
    save_pdvs,
    save_classes,
    save_agents,
)
from pdv_utils import apply_and_persist_pdv_adjustments

logger = logging.getLogger(__name__)

CHATLOG_DIR = "chatlogs"
SENT_MESSAGES_PATH = os.path.join(CHATLOG_DIR, "messages_to_humans.json")

_discord_queue: asyncio.Queue | None = None
_discord_client: discord.Client | None = None
_discord_task: asyncio.Task | None = None
_discord_consumer_task: asyncio.Task | None = None
_discord_loop: asyncio.AbstractEventLoop | None = None


def _get_conductor():
    import importlib
    return importlib.import_module("conductor")

def get_discord_queue() -> asyncio.Queue:
    global _discord_queue
    if _discord_queue is None:
        _discord_queue = asyncio.Queue()
    return _discord_queue


class _DiscordInUI(discord.Client):
    async def on_ready(self):
        print(f"[UI/Discord] Logged in as {self.user}")

    async def on_message(self, msg):
        channel_id_env = os.getenv("DISCORD_CHANNEL_ID")
        try:
            target_channel = int(channel_id_env) if channel_id_env else 0
        except Exception:
            target_channel = 0

        if msg.author.bot or (target_channel and msg.channel.id != target_channel):
            return

        author = getattr(msg.author, "display_name", str(msg.author))
        entry = {
            "source": "discord",
            "author": author,
            "text": msg.content,
            "timestamp": msg.created_at.isoformat(),
        }

        await get_discord_queue().put(entry)

        try:
            g = load_globals() or {}
            adjs = g.get("incoming_message_pdvms") or g.get("incoming_message_dpvms") or []
            if isinstance(adjs, list) and adjs:
                norm = []
                for item in adjs:
                    if "delta_pct" in item and "delta" not in item:
                        try:
                            pct = float(item["delta_pct"])
                            item = {**item, "delta": pct / 100.0}
                            item.pop("delta_pct", None)
                        except Exception:
                            continue
                    norm.append(item)
                if norm:
                    apply_and_persist_pdv_adjustments(norm)
                    # Best-effort push to keep conductor in sync immediately.
                    try:
                        c = _get_conductor()
                        if hasattr(c, "_refresh_pdvs_from_disk"):
                            c._refresh_pdvs_from_disk()
                    except Exception:
                        pass
                    print("[PDVM] Applied incoming_message_pdvms on Discord message.")
        except Exception as e:
            print(f"[PDVM] Failed applying incoming_message_pdvms: {e}")


async def _discord_consumer_loop():
    q = get_discord_queue()
    while True:
        item = await q.get()
        try:
            c = _get_conductor()
            if hasattr(c, "inject_external_message"):
                await c.inject_external_message(item.get("text", ""), item)
            else:
                await c.handle_user_message(item.get("text", ""), meta=item)
        except Exception as e:
            print(f"[UI/Discord] Consumer error: {e}")
        finally:
            q.task_done()



async def start_discord_in_ui():
    global _discord_client, _discord_task, _discord_consumer_task, _discord_loop

    token = os.getenv("fenrir_token")
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    if not token or not channel_id:
        print("[UI/Discord] Disabled (missing fenrir_token or DISCORD_CHANNEL_ID).")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    _discord_client = _DiscordInUI(intents=intents)

    _discord_loop = asyncio.get_running_loop()
    loop = _discord_loop
    # Queue removed: do not start consumer loop.

    async def _runner():
        try:
            await _discord_client.start(token)
        except Exception as e:
            print(f"[UI/Discord] Client stopped: {e}")

    _discord_task = loop.create_task(_runner())
    print("[UI/Discord] Listening started.")


async def stop_discord_in_ui():
    global _discord_client, _discord_task, _discord_consumer_task
    if _discord_client:
        try:
            await _discord_client.close()
        except Exception:
            pass
    if _discord_task:
        _discord_task.cancel()
    if _discord_consumer_task:
        _discord_consumer_task.cancel()
    print("[UI/Discord] Listening stopped.")

# ─── public: fetch recent discord messages for listeners ──────────────────────
async def _discord_fetch_recent(n: int) -> list[dict]:
    """Coroutine to fetch last n messages from configured channel."""
    if _discord_client is None:
        return []
    try:
        chan_id = int(os.getenv("DISCORD_CHANNEL_ID") or "0")
    except Exception:
        chan_id = 0
    if not chan_id:
        return []
    ch = _discord_client.get_channel(chan_id)
    if ch is None:
        try:
            ch = await _discord_client.fetch_channel(chan_id)
        except Exception:
            return []
    out: list[dict] = []
    async for m in ch.history(limit=int(n)):
        out.append({
            "author": getattr(m.author, "display_name", str(m.author)),
            "text": m.content,
            "timestamp": m.created_at.isoformat(),
        })
    return out

def fetch_recent_discord_messages(n: int = 10) -> list[dict]:
    """Sync wrapper to retrieve last n messages from Discord."""
    loop = _discord_loop
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return []
    fut = asyncio.run_coroutine_threadsafe(_discord_fetch_recent(int(n)), loop)
    try:
        return fut.result(timeout=5)
    except Exception:
        return []


def hsl_to_hex(h: int, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def pastel_for_class(name: str) -> str:
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % 360
    return hsl_to_hex(h, 0.45, 0.82)


class FenrirUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(self, agents, inject_callback=None, send_callback=None, config_path="confs/globals.json"):
        logger.debug(
            "Entering FenrirUI.__init__ with agents=%s inject_callback=%s send_callback=%s",
            agents,
            inject_callback,
            send_callback,
        )
        self.root = tk.Tk()
        self.root.title("Fenrir")
        self.agents = agents
        self.inject_callback = inject_callback
        self.send_callback = send_callback
        self.config_path = config_path

        self.sent_messages = []
        self.log_messages = []
        self._agent_payloads: dict[str, str] = {}
        self._active_agent: Optional[str] = None
        self._group_contexts: dict[str, str] = {}

        self._model_cache: list[str] = []

        self.global_config = load_globals()

        pdv_cfg = load_pdvs()
        self.pdv_values = {name: cfg.get("value", 0.0) for name, cfg in pdv_cfg.items()}

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ----- Configurations Tab -----
        configs_tab = ttk.Frame(self.notebook)
        self.notebook.add(configs_tab, text="Configurations")

        self.config_nb = ttk.Notebook(configs_tab)
        self.config_nb.pack(fill=tk.BOTH, expand=True)

        # Globals sub-tab
        self.globals_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.globals_tab, text="Globals")
        self._build_globals_tab()

        # PDVs sub-tab
        self.pdvs_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.pdvs_tab, text="PDVs")
        self._build_pdvs_tab()

        # Agent Classes sub-tab
        self.classes_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.classes_tab, text="Agent Classes")
        self._build_classes_tab()

        # Agents sub-tab
        self.agents_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.agents_tab, text="Agents")
        self._build_agents_tab()

        # ----- Live Metrics Tab -----
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="Live Metrics")

        self.metric_bars: dict[str, ttk.Progressbar] = {}
        self.metric_labels: dict[str, tk.Label] = {}
        for name in self.pdv_values.keys():
            frame = ttk.Frame(metrics_tab)
            frame.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(frame, text=name + ":", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
            bar = ttk.Progressbar(frame, maximum=100, mode="determinate")
            bar.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
            val = ttk.Label(frame, text="0.00")
            val.pack(side=tk.LEFT, padx=4)
            self.metric_bars[name] = bar
            self.metric_labels[name] = val

        limit = self.global_config.get("max_context_tokens", 8192)
        self.token_usage_var = tk.StringVar(value=f"Tokens: 0 / {limit}")
        tk.Label(metrics_tab, textvariable=self.token_usage_var).pack(anchor="w", padx=4, pady=2)

        # ----- Internal Thoughts Tab -----
        thoughts_tab = ttk.Frame(self.notebook)
        self.notebook.add(thoughts_tab, text="Internal Thoughts")

        paned = ttk.Panedwindow(thoughts_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        thought_frame = ttk.Frame(paned)
        event_frame = ttk.Frame(paned)
        paned.add(thought_frame, weight=1)
        paned.add(event_frame, weight=1)

        self.thought_stream = scrolledtext.ScrolledText(thought_frame, state="disabled")
        self.thought_stream.pack(fill=tk.BOTH, expand=True)

        self.events_stream = scrolledtext.ScrolledText(event_frame, state="disabled")
        self.events_stream.pack(fill=tk.BOTH, expand=True)

        # Backward compatibility
        self.output = self.thought_stream

        self.base_timeout = self.global_config.get("watchdog_timeout", 900)
        label_txt = (
            "Base Timeout: disabled"
            if (self.base_timeout is None or float(self.base_timeout) <= 0)
            else f"Base Timeout: {int(self.base_timeout)}s"
        )
        self.timeout_label = ttk.Label(thoughts_tab, text=label_txt)
        self.timeout_label.pack(anchor="w", padx=4, pady=2)

        self._refresh_log_display()

        # ----- Messages Tab -----
        messages_tab = ttk.Frame(self.notebook)
        self.notebook.add(messages_tab, text="Messages")

        msg_top = ttk.Frame(messages_tab)
        msg_top.pack(fill=tk.X, pady=2)
        ttk.Button(msg_top, text="Refresh", command=self.update_queue_and_sent).pack(
            side=tk.RIGHT, padx=2
        )

        queued_frame = ttk.LabelFrame(messages_tab, text="Queued (from humans)")
        queued_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.queued_text = scrolledtext.ScrolledText(queued_frame, state="disabled", height=10)
        self.queued_text.pack(fill=tk.BOTH, expand=True)

        sent_frame = ttk.LabelFrame(messages_tab, text="Sent (to humans)")
        sent_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.sent_text = scrolledtext.ScrolledText(sent_frame, state="disabled", height=10)
        self.sent_text.pack(fill=tk.BOTH, expand=True)

        self.update_queue_and_sent()

        # ----- Topology Tab -----
        topology_tab = ttk.Frame(self.notebook)
        self.notebook.add(topology_tab, text="Topology")

        self.topology_header = ttk.Label(topology_tab, text="Active Agent: None")
        self.topology_header.pack(anchor="w", padx=4, pady=2)

        self.topology_canvas = tk.Canvas(topology_tab, background="white")
        self.topology_canvas.pack(fill=tk.BOTH, expand=True)
        self.topology_canvas.bind("<Configure>", lambda e: self._redraw_topology())

        self._topology_active = None
        self._topology_agents = []
        self._topology_node_items = {}
        self._topology_tooltip = None

        # ----- Agent Context Tab -----
        agent_tab = ttk.Frame(self.notebook)
        self.notebook.add(agent_tab, text="Agent Context")

        self.agent_header = ttk.Label(agent_tab, text="Current Agent: None")
        self.agent_header.pack(anchor="w", padx=4, pady=2)

        agent_toolbar = ttk.Frame(agent_tab)
        agent_toolbar.pack(anchor="w", padx=4, pady=2)
        ttk.Button(agent_toolbar, text="Copy JSON", command=self._copy_agent_payload).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(agent_toolbar, text="Save…", command=self._save_agent_payload).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(agent_toolbar, text="Clear", command=self._clear_agent_payload).pack(
            side=tk.LEFT, padx=2
        )

        self.agent_payload_view = scrolledtext.ScrolledText(
            agent_tab, state="disabled"
        )
        self.agent_payload_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # ----- Group Context Tab -----
        group_tab = ttk.Frame(self.notebook)
        self.notebook.add(group_tab, text="Group Context")

        group_paned = ttk.Panedwindow(group_tab, orient=tk.HORIZONTAL)
        group_paned.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(group_paned)
        text_frame = ttk.Frame(group_paned)
        group_paned.add(list_frame, weight=1)
        group_paned.add(text_frame, weight=3)

        self.group_list = tk.Listbox(list_frame, exportselection=False)
        self.group_list.pack(fill=tk.BOTH, expand=True)
        self.group_list.bind("<<ListboxSelect>>", self._on_group_select)

        self.group_text = scrolledtext.ScrolledText(text_frame, state="disabled")
        self.group_text.pack(fill=tk.BOTH, expand=True)

        self.update_pdvs(self.pdv_values)
        self._start_metrics_poll()
        self._ensure_globals_set()
        logger.debug("Exiting FenrirUI.__init__")


    class _InjectDialog(simpledialog.Dialog):
        """Dialog for entering a message to inject."""

        def __init__(self, parent, group_name: str):
            logger.debug("Entering _InjectDialog.__init__ group_name=%s", group_name)
            self.group_name = group_name
            self.message = ""
            super().__init__(parent, title="Inject Message")
            logger.debug("Exiting _InjectDialog.__init__")

        def body(self, master):
            logger.debug("Entering _InjectDialog.body")
            tk.Label(master, text=f"Send message to {self.group_name}:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _InjectDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _InjectDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            self.result = self.message
            logger.debug("Exiting _InjectDialog.apply")

    class _SendDialog(simpledialog.Dialog):
        """Dialog for entering a message for the listeners."""

        def body(self, master):
            logger.debug("Entering _SendDialog.body")
            tk.Label(master, text="Message to user:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            tk.Label(master, text="Groups (comma-separated):").grid(row=2, column=0, sticky="w")
            self.groups_entry = tk.Entry(master)
            self.groups_entry.grid(row=3, column=0, sticky="ew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _SendDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _SendDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            groups_text = self.groups_entry.get().strip()
            groups = [g.strip() for g in groups_text.split(",") if g.strip()]
            self.result = {"message": self.message, "groups": groups}
        logger.debug("Exiting _SendDialog.apply")

    def _inject_message(self):
        logger.debug("Entering _inject_message")
        group_name = "All Groups"
        dialog = self._InjectDialog(self.root, group_name)
        result = dialog.result
        if result:
            if self.inject_callback:
                self.inject_callback(group_name, result)
            else:
                items = self._enqueue_message("system", result)
                self.update_queue(items)
                try:
                    c = _get_conductor()
                    import asyncio
                    asyncio.run(c.inject_external_message(result, {"author": "system"}))
                except Exception as e:
                    print(f"[UI] inject_external_message failed: {e}")
        logger.debug("Exiting _inject_message")

    def _send_message(self):
        logger.debug("Entering _send_message")
        dialog = self._SendDialog(self.root)
        result = dialog.result
        if result:
            groups = result.get("groups") if isinstance(result, dict) else []
            if not groups:
                messagebox.showerror("Error", "Please specify at least one group")
            else:
                if self.send_callback:
                    self.send_callback(result["message"], groups)
                else:
                    items = self._enqueue_message("user", result["message"])
                    self.update_queue(items)
                    try:
                        c = _get_conductor()
                        import asyncio
                        asyncio.run(
                            c.inject_external_message(result["message"], {"author": "user", "groups": groups})
                        )
                    except Exception as e:
                        print(f"[UI] inject_external_message failed: {e}")
        logger.debug("Exiting _send_message")

    def update_queue(self, messages):
        logger.debug("Entering update_queue messages=%s", messages)
        self.update_queue_and_sent(queued=messages)
        logger.debug("Exiting update_queue")

    def update_sent(self, messages):
        logger.debug("Entering update_sent messages=%s", messages)
        self.sent_messages = list(messages)
        self.update_queue_and_sent(sent=messages)
        logger.debug("Exiting update_sent")

    def update_pdvs(self, pdv_values: dict[str, float]) -> None:
        logger.debug("Entering update_pdvs pdv_values=%s", pdv_values)

        def _update():
            for name, value in pdv_values.items():
                bar = self.metric_bars.get(name)
                label = self.metric_labels.get(name)
                if bar and label:
                    pct = max(0.0, min(100.0, value * 100.0))
                    bar["value"] = pct
                    label.config(text=f"{value:.2f}")

        self._threadsafe(_update)
        logger.debug("Exiting update_pdvs")

    def set_token_usage(self, used: int, limit: int) -> None:
        def _update():
            self.token_usage_var.set(f"Tokens: {used} / {limit}")
        self._threadsafe(_update)

    def append_thought(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_thought text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.thought_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_thought")

    def append_event(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_event text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.events_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_event")

    def update_queue_and_sent(self, queued: Optional[list] = None, sent: Optional[list] = None) -> None:
        logger.debug("Entering update_queue_and_sent queued=%s sent=%s", queued, sent)

        def _load(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return None

        if queued is None:
            try:
                c = _get_conductor()
                queued = getattr(c, "_INCOMING_QUEUE", [])
            except Exception:
                queued = []
        if sent is None:
            sent = _load(SENT_MESSAGES_PATH)

        def _render():
            # Queued messages
            self.queued_text.configure(state="normal")
            self.queued_text.delete("1.0", tk.END)
            if isinstance(queued, list) and queued:
                for entry in queued:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        msg = entry.get("raw_message") or entry.get("message") or str(entry)
                    else:
                        ts = "unknown"
                        msg = str(entry)
                    self.queued_text.insert(tk.END, f"[{ts}] {msg}\n")
            else:
                self.queued_text.insert(tk.END, "No queued messages.\n")
            self.queued_text.configure(state="disabled")
            self.queued_text.see(tk.END)

            # Sent messages
            self.sent_text.configure(state="normal")
            self.sent_text.delete("1.0", tk.END)
            if isinstance(sent, list) and sent:
                for entry in sent:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        sender = entry.get("sender", "unknown")
                        msg = entry.get("message", "")
                    else:
                        ts = "unknown"
                        sender = "unknown"
                        msg = str(entry)
                    self.sent_text.insert(tk.END, f"[{ts}] {sender}: {msg}\n")
            else:
                self.sent_text.insert(tk.END, "No sent messages.\n")
            self.sent_text.configure(state="disabled")
            self.sent_text.see(tk.END)

        self._threadsafe(_render)
        logger.debug("Exiting update_queue_and_sent")

    # ------------------------------------------------------------------
    # Configuration tab builders
    # ------------------------------------------------------------------

    def _build_globals_tab(self) -> None:
        for child in self.globals_tab.winfo_children():
            child.destroy()
        models = self._fetch_models()
        self._globals_vars = {
            "debug_level": tk.StringVar(value=self.global_config.get("debug_level", "INFO")),
            "model": tk.StringVar(value=self.global_config.get("model", "")),
            "temperature": tk.StringVar(value=str(self.global_config.get("temperature", ""))),
            "system_prompt": tk.StringVar(value=self.global_config.get("system_prompt", "")),
            "pre_context_message": tk.StringVar(value=self.global_config.get("pre_context_message", "")),
            "post_context_message": tk.StringVar(value=self.global_config.get("post_context_message", "")),
            "max_context_tokens": tk.StringVar(value=str(self.global_config.get("max_context_tokens", 8192))),
            "pdv_gamma": tk.StringVar(value=str(self.global_config.get("pdv_gamma", 2.0))),
            "watchdog_timeout": tk.StringVar(value=str(self.global_config.get("watchdog_timeout", 900))),
        }
        row = 0
        for label, key in [
            ("Debug Level", "debug_level"),
            ("Model", "model"),
            ("Temperature", "temperature"),
            ("System Prompt", "system_prompt"),
            ("Pre Context", "pre_context_message"),
            ("Post Context", "post_context_message"),
            ("Max Tokens", "max_context_tokens"),
            ("PDV Gamma", "pdv_gamma"),
            ("Watchdog Timeout (s)  (0=disabled)", "watchdog_timeout"),
        ]:
            tk.Label(self.globals_tab, text=label).grid(row=row, column=0, sticky="w")
            if key == "model":
                box = ttk.Combobox(
                    self.globals_tab,
                    textvariable=self._globals_vars[key],
                    values=models,
                    state="readonly",
                )
                box.grid(row=row, column=1, sticky="ew")
            else:
                entry = ttk.Entry(self.globals_tab, textvariable=self._globals_vars[key])
                entry.grid(row=row, column=1, sticky="ew")
            row += 1
        self.globals_tab.columnconfigure(1, weight=1)
        btn_frame = ttk.Frame(self.globals_tab)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=4)
        ttk.Button(btn_frame, text="Refresh Models", command=self._refresh_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Save", command=self._save_globals).pack(side=tk.LEFT, padx=2)

    def _refresh_models(self) -> None:
        models = self._fetch_models()
        box = None
        for child in self.globals_tab.grid_slaves():
            if isinstance(child, ttk.Combobox):
                box = child
                break
        if box is not None:
            box.configure(values=models)

    def _save_globals(self) -> None:
        for k, var in self._globals_vars.items():
            val = var.get()
            if k in {"temperature", "max_context_tokens", "pdv_gamma", "watchdog_timeout"}:
                try:
                    if k == "max_context_tokens":
                        self.global_config[k] = int(val)
                    elif k == "watchdog_timeout":
                        self.global_config[k] = int(float(val))
                    else:
                        self.global_config[k] = float(val)
                except ValueError:
                    self.global_config[k] = None
            else:
                self.global_config[k] = val
        save_globals(self.global_config)
        # refresh label
        self.base_timeout = self.global_config.get("watchdog_timeout", 900)
        txt = (
            "Base Timeout: disabled"
            if (self.base_timeout is None or float(self.base_timeout) <= 0)
            else f"Base Timeout: {int(self.base_timeout)}s"
        )
        self.timeout_label.config(text=txt)

    def _build_pdvs_tab(self) -> None:
        for child in self.pdvs_tab.winfo_children():
            child.destroy()
        self._pdv_rows = []
        pdvs = load_pdvs()
        frame = self.pdvs_tab
        for name, cfg in pdvs.items():
            self._add_pdv_row(frame, name, cfg)
        btn = ttk.Frame(frame)
        btn.pack(fill=tk.X, pady=4)
        ttk.Button(btn, text="Add", command=lambda: self._add_pdv_row(frame, "new", {"description": "", "value": 0.0})).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Save", command=self._save_pdvs).pack(side=tk.LEFT, padx=2)

    def _add_pdv_row(self, parent, name, cfg):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=2)
        name_var = tk.StringVar(value=name)
        desc_var = tk.StringVar(value=cfg.get("description", ""))
        val_var = tk.DoubleVar(value=cfg.get("value", 0.0))
        ttk.Entry(row, textvariable=name_var, width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=desc_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Scale(row, from_=0.0, to=1.0, variable=val_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Remove", command=lambda r=row: r.destroy()).pack(side=tk.LEFT, padx=2)
        self._pdv_rows.append((row, name_var, desc_var, val_var))

    def _save_pdvs(self) -> None:
        data = {}
        for row, name_var, desc_var, val_var in self._pdv_rows:
            if not row.winfo_exists():
                continue
            name = name_var.get().strip()
            if not name:
                continue
            data[name] = {
                "name": name,
                "description": desc_var.get(),
                "value": float(val_var.get()),
            }
        save_pdvs(data)
        self.pdv_values = {n: cfg["value"] for n, cfg in data.items()}
        self.update_pdvs(self.pdv_values)

    def _build_classes_tab(self) -> None:
        for child in self.classes_tab.winfo_children():
            child.destroy()
        classes = load_classes()
        text = scrolledtext.ScrolledText(self.classes_tab)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", json.dumps(list(classes.values()), indent=2))
        btn = ttk.Frame(self.classes_tab)
        btn.pack(fill=tk.X)
        def _save():
            try:
                data = json.loads(text.get("1.0", tk.END))
                cls_map = {c["name"]: c for c in data}
                save_classes(cls_map)
            except Exception:
                messagebox.showerror("Error", "Invalid JSON")
        ttk.Button(btn, text="Save", command=_save).pack(side=tk.RIGHT, padx=2)

    def _build_agents_tab(self) -> None:
        for child in self.agents_tab.winfo_children():
            child.destroy()
        agents = load_agents()
        self._agents_cache = agents
        flagged = [a for a in agents if a.get("flag_no_downstream")]
        if flagged:
            banner = ttk.Frame(self.agents_tab, relief=tk.RIDGE, borderwidth=1)
            banner.pack(fill=tk.X, pady=2)
            top = ttk.Frame(banner)
            top.pack(fill=tk.X)
            ttk.Label(top, text="Agents with no downstream:", foreground="red").pack(side=tk.LEFT)
            ttk.Button(top, text="Dismiss", command=banner.destroy).pack(side=tk.RIGHT)
            self._flag_list = tk.Listbox(banner, height=min(5, len(flagged)), exportselection=False)
            for a in flagged:
                self._flag_list.insert(tk.END, a["name"])
            self._flag_list.pack(fill=tk.X)
            self._flag_list.bind("<<ListboxSelect>>", self._on_flag_select)
            self._flag_info = tk.Label(banner, justify="left")
            self._flag_info.pack(fill=tk.X, padx=4, pady=2)
            btns = ttk.Frame(banner)
            btns.pack(fill=tk.X, pady=4)
            ttk.Button(btns, text="Batch Wiring", command=self._batch_wiring).pack(side=tk.LEFT, padx=2)
            ttk.Button(btns, text="Repair Dead-Ends", command=self._repair_dead_ends).pack(side=tk.LEFT, padx=2)

        text = scrolledtext.ScrolledText(self.agents_tab)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", json.dumps(agents, indent=2))
        btn = ttk.Frame(self.agents_tab)
        btn.pack(fill=tk.X)

        def _save() -> None:
            try:
                data = json.loads(text.get("1.0", tk.END))
                save_agents(data)
                self._build_agents_tab()
            except Exception:
                messagebox.showerror("Error", "Invalid JSON")

        ttk.Button(btn, text="Save", command=_save).pack(side=tk.RIGHT, padx=2)

    def _on_flag_select(self, _event=None) -> None:
        sel = getattr(self, "_flag_list", None)
        if not sel:
            return
        idx = sel.curselection()
        if not idx:
            return
        name = sel.get(idx[0])
        agent = next((a for a in self._agents_cache if a["name"] == name), None)
        if not agent:
            return
        lines = []
        for g in agent.get("groups_out", []):
            consumers = [b["name"] for b in self._agents_cache if b["name"] != name and g in b.get("groups_in", [])]
            if consumers:
                lines.append(f"{g} -> {', '.join(consumers)}")
            else:
                lines.append(f"{g} -> (no consumers)")
        self._flag_info.config(text="\n".join(lines))

    def _batch_wiring(self) -> None:
        if not hasattr(self, "_flag_list"):
            return
        idx = self._flag_list.curselection()
        if not idx:
            messagebox.showinfo("Batch Wiring", "Select a flagged agent first")
            return
        name = self._flag_list.get(idx[0])
        agent = next(a for a in self._agents_cache if a["name"] == name)
        missing = agent.get("missing_out_groups") or [
            g for g in agent.get("groups_out", [])
            if not any(g in b.get("groups_in", []) for b in self._agents_cache if b["name"] != name)
        ]
        if not missing:
            messagebox.showinfo("Batch Wiring", "No missing groups")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Wiring")
        selections = []
        for g in missing:
            ttk.Label(dialog, text=f"{g}:").pack(anchor="w")
            cands = [b for b in self._agents_cache if b["name"] != name and g not in b.get("groups_in", [])]
            for cand in cands:
                var = tk.BooleanVar()
                chk = tk.Checkbutton(dialog, text=cand["name"], variable=var)
                chk.pack(anchor="w", padx=20)
                selections.append((var, cand["name"], g))
        def _apply() -> None:
            import copy, json
            new_agents = copy.deepcopy(self._agents_cache)
            affected = set()
            for var, cand_name, grp in selections:
                if var.get():
                    targ = next(a for a in new_agents if a["name"] == cand_name)
                    targ.setdefault("groups_in", [])
                    if grp not in targ["groups_in"]:
                        targ["groups_in"].append(grp)
                        affected.add(cand_name)
            if not affected:
                dialog.destroy()
                return
            targ_agent = next(a for a in new_agents if a["name"] == name)
            targ_agent["flag_no_downstream"] = False
            targ_agent["missing_out_groups"] = []
            affected.add(name)
            preview = {a["name"]: a for a in new_agents if a["name"] in affected}
            prev_win = tk.Toplevel(dialog)
            prev_win.title("Preview Changes")
            txt = scrolledtext.ScrolledText(prev_win)
            txt.pack(fill=tk.BOTH, expand=True)
            txt.insert("1.0", json.dumps(preview, indent=2))
            def _confirm() -> None:
                save_agents(new_agents)
                prev_win.destroy()
                dialog.destroy()
                self._build_agents_tab()
            ttk.Button(prev_win, text="Apply", command=_confirm).pack()
        ttk.Button(dialog, text="Apply", command=_apply).pack(pady=4)

    def _repair_dead_ends(self) -> None:
        import copy, json
        new_agents = copy.deepcopy(self._agents_cache)
        affected = set()
        for agent in new_agents:
            if agent.get("flag_no_downstream"):
                missing = agent.get("missing_out_groups") or [
                    g for g in agent.get("groups_out", [])
                    if not any(g in b.get("groups_in", []) for b in new_agents if b["name"] != agent["name"])
                ]
                for g in missing:
                    cands = [b for b in new_agents if b["name"] != agent["name"] and g not in b.get("groups_in", [])]
                    if not cands:
                        continue
                    targ = sorted(cands, key=lambda x: x["name"])[0]
                    targ.setdefault("groups_in", [])
                    if g not in targ["groups_in"]:
                        targ["groups_in"].append(g)
                        affected.add(targ["name"])
                agent["flag_no_downstream"] = False
                agent["missing_out_groups"] = []
                affected.add(agent["name"])
        if not affected:
            messagebox.showinfo("Repair Dead-Ends", "No changes proposed")
            return
        preview = {a["name"]: a for a in new_agents if a["name"] in affected}
        win = tk.Toplevel(self.root)
        win.title("Preview Repair")
        txt = scrolledtext.ScrolledText(win)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", json.dumps(preview, indent=2))
        def _apply() -> None:
            save_agents(new_agents)
            win.destroy()
            self._build_agents_tab()
        ttk.Button(win, text="Apply", command=_apply).pack()

    def _fetch_models(self) -> list[str]:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            self._model_cache = sorted([m for m in models if m])
        except Exception as exc:
            logger.exception("Model list fetch failed")
            messagebox.showwarning("Models", f"Failed to fetch models: {exc}")
        return list(self._model_cache)

    def _ensure_globals_set(self) -> None:
        need = not self.global_config.get("model") or (
            self.global_config.get("temperature") is None
        )
        if need:
            self._open_globals_dialog_blocking()

    def _open_globals_dialog_blocking(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Set Globals")
        dlg.transient(self.root)
        vars = {
            "debug_level": tk.StringVar(value=self.global_config.get("debug_level", "INFO")),
            "model": tk.StringVar(value=self.global_config.get("model", "")),
            "temperature": tk.StringVar(value=str(self.global_config.get("temperature", ""))),
            "system_prompt": tk.StringVar(value=self.global_config.get("system_prompt", "")),
            "pre_context_message": tk.StringVar(value=self.global_config.get("pre_context_message", "")),
            "post_context_message": tk.StringVar(value=self.global_config.get("post_context_message", "")),
            "max_context_tokens": tk.StringVar(value=str(self.global_config.get("max_context_tokens", 8192))),
            "pdv_gamma": tk.StringVar(value=str(self.global_config.get("pdv_gamma", 2.0))),
            "watchdog_timeout": tk.StringVar(value=str(self.global_config.get("watchdog_timeout", 900))),
        }
        models = self._fetch_models()
        row = 0
        for label, key in [
            ("Debug Level", "debug_level"),
            ("Model", "model"),
            ("Temperature", "temperature"),
            ("System Prompt", "system_prompt"),
            ("Pre Context", "pre_context_message"),
            ("Post Context", "post_context_message"),
            ("Max Tokens", "max_context_tokens"),
            ("PDV Gamma", "pdv_gamma"),
            ("Watchdog Timeout (s)  (0=disabled)", "watchdog_timeout"),
        ]:
            tk.Label(dlg, text=label).grid(row=row, column=0, sticky="w")
            if key == "model":
                box = ttk.Combobox(dlg, textvariable=vars[key], values=models, state="readonly")
                box.grid(row=row, column=1, sticky="ew")
            else:
                entry = ttk.Entry(dlg, textvariable=vars[key])
                entry.grid(row=row, column=1, sticky="ew")
            row += 1
        dlg.columnconfigure(1, weight=1)
        btn = ttk.Frame(dlg)
        btn.grid(row=row, column=0, columnspan=2, pady=4)

        def _refresh() -> None:
            box.configure(values=self._fetch_models())

        def _save() -> None:
            temp_cfg = {}
            for k, var in vars.items():
                val = var.get()
                if k in {"temperature", "max_context_tokens", "pdv_gamma", "watchdog_timeout"}:
                    try:
                        if k == "max_context_tokens":
                            temp_cfg[k] = int(val)
                        elif k == "watchdog_timeout":
                            temp_cfg[k] = int(float(val))
                        else:
                            temp_cfg[k] = float(val)
                    except ValueError:
                        messagebox.showerror("Globals", f"Invalid value for {k}")
                        return
                else:
                    temp_cfg[k] = val
            if not temp_cfg.get("model") or temp_cfg.get("temperature") is None:
                messagebox.showerror(
                    "Globals", "Model and temperature are required and must be valid"
                )
                return
            self.global_config.update(temp_cfg)
            save_globals(self.global_config)
            self.global_config = load_globals()
            self._build_globals_tab()
            self.base_timeout = self.global_config.get("watchdog_timeout", 900)
            txt = (
                "Base Timeout: disabled"
                if (self.base_timeout is None or float(self.base_timeout) <= 0)
                else f"Base Timeout: {int(self.base_timeout)}s"
            )
            self.timeout_label.config(text=txt)
            dlg.grab_release()
            dlg.destroy()

        ttk.Button(btn, text="Refresh Models", command=_refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Save", command=_save).pack(side=tk.LEFT, padx=2)

        dlg.grab_set()
        self.root.wait_window(dlg)

    def _start_metrics_poll(self):
        def _tick():
            try:
                with open(
                    os.path.join("chatlogs", "pdvs_live.json"), "r", encoding="utf-8"
                ) as f:
                    pdv_vals = json.load(f)
                if isinstance(pdv_vals, dict):
                    self.update_pdvs(pdv_vals)
            except Exception:
                pass
            try:
                with open(
                    os.path.join("chatlogs", "token_usage.json"), "r", encoding="utf-8"
                ) as f:
                    info = json.load(f)
                self.set_token_usage(int(info.get("used", 0)), int(info.get("limit", 0)))
            except Exception:
                pass
            self.root.after(2000, _tick)

        self.root.after(2000, _tick)


    def set_active_agent(self, name: str) -> None:
        logger.debug("Entering set_active_agent name=%s", name)

        def _update():
            self._active_agent = name or None
            display = name or "None"
            self.agent_header.config(text=f"Current Agent: {display}")
            payload = self._agent_payloads.get(name) if name else ""
            self._render_agent_payload(payload)

        self._threadsafe(_update)
        logger.debug("Exiting set_active_agent")

    def update_agent_payload(self, agent: str, payload: dict) -> None:
        logger.debug(
            "Entering update_agent_payload agent=%s keys=%s",
            agent,
            list(payload.keys()),
        )
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self._agent_payloads[agent] = text

        def _update():
            if self._active_agent == agent:
                self._render_agent_payload(text)

        self._threadsafe(_update)
        logger.debug("Exiting update_agent_payload")

    def set_group_contexts(self, context_by_group: dict[str, str]) -> None:
        logger.debug(
            "Entering set_group_contexts context_by_group_keys=%s",
            list(context_by_group.keys()),
        )

        def _update():
            current = None
            sel = self.group_list.curselection()
            if sel:
                current = self.group_list.get(sel[0])
            self._group_contexts = dict(context_by_group)
            self.group_list.delete(0, tk.END)
            for grp in self._group_contexts:
                self.group_list.insert(tk.END, grp)
            target = current if current in self._group_contexts else None
            if not target and self._group_contexts:
                target = next(iter(self._group_contexts))
                idx = list(self._group_contexts).index(target)
                self.group_list.selection_set(idx)
            if target:
                self._render_group_text(self._group_contexts.get(target, ""))
            else:
                self._render_group_text("")

        self._threadsafe(_update)
        logger.debug("Exiting set_group_contexts")

    def update_group_context(self, group: str, text: str) -> None:
        logger.debug("Entering update_group_context group=%s", group)

        def _update():
            self._group_contexts[group] = text
            names = list(self.group_list.get(0, tk.END))
            if group not in names:
                self.group_list.insert(tk.END, group)
            sel = self.group_list.curselection()
            if sel and self.group_list.get(sel[0]) == group:
                self._render_group_text(text)

        self._threadsafe(_update)
        logger.debug("Exiting update_group_context")

    def _on_group_select(self, _event=None):
        sel = self.group_list.curselection()
        if not sel:
            return
        group = self.group_list.get(sel[0])
        text = self._group_contexts.get(group, "")
        self._render_group_text(text)

    def _render_agent_payload(self, text: str | None) -> None:
        self.agent_payload_view.configure(state="normal")
        self.agent_payload_view.delete("1.0", tk.END)
        if text:
            self.agent_payload_view.insert(tk.END, text)
            lines = self.agent_payload_view.get("1.0", tk.END).splitlines()
            if len(lines) > 2000:
                trimmed = "\n".join(lines[-2000:])
                self.agent_payload_view.delete("1.0", tk.END)
                self.agent_payload_view.insert(tk.END, trimmed)
        self.agent_payload_view.configure(state="disabled")
        self.agent_payload_view.see(tk.END)

    def _render_group_text(self, text: str) -> None:
        self.group_text.configure(state="normal")
        self.group_text.delete("1.0", tk.END)
        if text:
            self.group_text.insert(tk.END, text)
        self.group_text.configure(state="disabled")
        self.group_text.see(tk.END)

    def _copy_agent_payload(self) -> None:
        logger.debug("Entering _copy_agent_payload")
        text = self.agent_payload_view.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        logger.debug("Exiting _copy_agent_payload")

    def _save_agent_payload(self) -> None:
        logger.debug("Entering _save_agent_payload")
        text = self.agent_payload_view.get("1.0", tk.END)
        if not text.strip():
            logger.debug("_save_agent_payload called with empty text")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if fname:
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as exc:  # noqa: BLE001
                logger.error("Failed to save JSON payload: %s", exc)
        logger.debug("Exiting _save_agent_payload")

    def _clear_agent_payload(self) -> None:
        logger.debug("Entering _clear_agent_payload")
        self._render_agent_payload("")
        logger.debug("Exiting _clear_agent_payload")

    def update_topology(self, active_agent: dict, agents: list[dict]) -> None:
        logger.debug("Entering update_topology active_agent=%s agents=%s", active_agent, agents)

        def _update():
            self._topology_active = active_agent
            self._topology_agents = agents
            self._redraw_topology()

        self._threadsafe(_update)
        logger.debug("Exiting update_topology")

    def _compute_neighbors(self, active: dict, agents: list[dict]) -> tuple[list[dict], list[dict], int, int]:
        upstream: list[dict] = []
        downstream: list[dict] = []
        active_in = set(active.get("groups_in", []) or [])
        active_out = set(active.get("groups_out", []) or [])
        for ag in agents:
            if ag.get("name") == active.get("name"):
                continue
            ag_in = set(ag.get("groups_in", []) or [])
            ag_out = set(ag.get("groups_out", []) or [])
            if ag_out & active_in:
                upstream.append(ag)
            if active_out & ag_in:
                downstream.append(ag)
        extra_up = max(0, len(upstream) - 25)
        extra_down = max(0, len(downstream) - 25)
        return upstream[:25], downstream[:25], extra_up, extra_down

    def _redraw_topology(self) -> None:
        active = self._topology_active
        self._hide_tooltip()
        self.topology_canvas.delete("all")
        if not active:
            self.topology_header.config(text="Active Agent: None")
            return
        cls = active.get('agent_class') or active.get('role', '')
        self.topology_header.config(
            text=f"Active Agent: {active.get('name')} ({cls.title()})"
        )
        agents = self._topology_agents
        upstream, downstream, extra_up, extra_down = self._compute_neighbors(active, agents)
        width = self.topology_canvas.winfo_width() or 1
        height = self.topology_canvas.winfo_height() or 1
        cx_up, cx_act, cx_down = width * 0.2, width * 0.5, width * 0.8
        active_y = height / 2

        def positions(n: int) -> list[float]:
            pad = 40
            if n <= 0:
                return []
            step = (height - pad * 2) / n
            return [pad + step / 2 + i * step for i in range(n)]

        up_pos = positions(len(upstream))
        down_pos = positions(len(downstream))
        self._topology_node_items.clear()

        for ag, y in zip(upstream, up_pos):
            self._draw_node(cx_up, y, ag, 16)
            self._draw_arrow(cx_up, y, cx_act, active_y)
        if extra_up:
            self.topology_canvas.create_text(cx_up, height - 20, text=f"+{extra_up} more…")
        elif not upstream:
            self.topology_canvas.create_text(
                cx_up, active_y, text="No likely sources", fill="#888888"
            )

        for ag, y in zip(downstream, down_pos):
            self._draw_node(cx_down, y, ag, 16)
            self._draw_arrow(cx_act, active_y, cx_down, y)
        if extra_down:
            self.topology_canvas.create_text(cx_down, height - 20, text=f"+{extra_down} more…")
        elif not downstream:
            self.topology_canvas.create_text(
                cx_down, active_y, text="No likely targets", fill="#888888"
            )

        self._draw_node(cx_act, active_y, active, 24)

    def _draw_node(self, x: float, y: float, agent: dict, radius: int) -> None:
        cls_name = agent.get("agent_class") or agent.get("role", "")
        color = pastel_for_class(cls_name)
        circle = self.topology_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="black",
        )
        name = agent.get("name", "")
        display = name if len(name) <= 18 else name[:17] + "…"
        text = self.topology_canvas.create_text(x, y + radius + 12, text=display)
        for item in (circle, text):
            self.topology_canvas.tag_bind(
                item,
                "<Enter>",
                lambda e, a=agent: self._show_tooltip(e.x_root, e.y_root, a),
            )
            self.topology_canvas.tag_bind(item, "<Leave>", lambda e: self._hide_tooltip())
            self.topology_canvas.tag_bind(
                item,
                "<Double-1>",
                lambda e, a=agent: self.update_topology(a, self._topology_agents),
            )
            self._topology_node_items[item] = agent

    def _draw_arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.topology_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST)

    def _show_tooltip(self, x: int, y: int, agent: dict) -> None:
        self._hide_tooltip()
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x + 10}+{y + 10}")
        info = (
            f"{agent.get('name')}\nRole: {agent.get('role')}\n"
            f"in: {len(agent.get('groups_in', []))}  out: {len(agent.get('groups_out', []))}"
        )
        ttk.Label(tip, text=info, relief=tk.SOLID, borderwidth=1, padding=2).pack()
        self._topology_tooltip = tip

    def _hide_tooltip(self) -> None:
        if self._topology_tooltip:
            self._topology_tooltip.destroy()
            self._topology_tooltip = None

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _threadsafe(self, func, *args, **kwargs) -> None:
        if threading.current_thread() is threading.main_thread():
            func(*args, **kwargs)
        else:
            self.root.after(0, lambda: func(*args, **kwargs))

    def _expand_all(self):
        logger.debug("_expand_all called but tree view removed")

    def _collapse_all(self):
        logger.debug("_collapse_all called but tree view removed")

    def _send_from_box(self):
        logger.debug("_send_from_box called but message box removed")

    def _refresh_chat_display(self):
        logger.debug("_refresh_chat_display called but chat display removed")

    def _refresh_log_display(self):
        logger.debug("Entering _refresh_log_display")

        def _update():
            self.thought_stream.configure(state="normal")
            self.thought_stream.delete("1.0", tk.END)
            for m in self.log_messages:
                text = f"[{m['timestamp']}] {m['sender']}: {m['message']}\n{'-'*80}\n\n"
                self.thought_stream.insert(tk.END, text)
            self.thought_stream.configure(state="disabled")
            self.thought_stream.see(tk.END)

        self._threadsafe(_update)
        logger.debug("Exiting _refresh_log_display")

    def log(self, entry):
        logger.debug("Entering log entry=%s", entry)
        self.log_messages.append(entry)
        text = f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n{'-'*80}\n\n"
        self._threadsafe(self._append_text, self.thought_stream, text)
        logger.debug("Exiting log")

    def start(self):
        logger.debug("Entering start")
        loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_discord_in_ui())
            loop.run_forever()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        try:
            self.root.mainloop()
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: asyncio.create_task(stop_discord_in_ui()))
                loop.call_soon_threadsafe(loop.stop)
            t.join()
        logger.debug("Exiting start")
