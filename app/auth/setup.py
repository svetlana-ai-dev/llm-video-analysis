"""Run from project root: python -m app.auth.setup (writes only auth settings)."""
from getpass import getpass
from getpass import GetPassWarning
from pathlib import Path
import argparse
import os
import sys
import warnings


TERMINAL_HELP = (
    "Скрытый ввод недоступен в этом терминале. Пароль не считан. "
    "Запустите python -m app.auth.setup --gui или используйте отдельное окно PowerShell "
    "без перенаправления ввода."
)


def _terminal_available():
    return sys.stdin is sys.__stdin__ and sys.stdin.isatty()


def _hidden_input(prompt):
    if not _terminal_available():
        raise SystemExit(TERMINAL_HELP)
    try:
        with warnings.catch_warnings():
            # Stop before getpass's fallback can echo a password to the terminal.
            warnings.simplefilter("error", GetPassWarning)
            return getpass(prompt)
    except (GetPassWarning, EOFError, OSError):
        raise SystemExit(TERMINAL_HELP) from None


def _validate_password(password, confirmation=None):
    if not password:
        raise SystemExit("Получен пустой пароль. Повторите ввод или используйте --gui.")
    if len(password) < 12:
        raise SystemExit("Пароль слишком короткий: требуется минимум 12 символов.")
    if len(password) > 1024:
        raise SystemExit("Пароль слишком длинный: допускается максимум 1024 символа.")
    if confirmation is not None and password != confirmation:
        raise SystemExit(
            "Пароли не совпадают. Повторите оба ввода с одной раскладкой клавиатуры "
            "или используйте --gui."
        )


def _validate_username(username):
    if not username or len(username) > 128 or any(c in username for c in "\r\n'\"\\#"):
        raise SystemExit("Укажите логин до 128 символов без кавычек, # и обратной косой черты.")


def _gui_credentials():
    # Native Unicode entries avoid terminal/ConPTY input, while masking passwords.
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
    except Exception:
        # Do not include exception content, which could contain input data.
        raise SystemExit("Не удалось открыть окно настройки. Используйте Python с Tkinter или отдельное окно PowerShell.") from None
    try:
        username = simpledialog.askstring("Настройка администратора", "Логин администратора:", parent=root)
        if username is None:
            raise SystemExit("Настройка отменена. Файл .env не изменён.")
        username = username.strip()
        _validate_username(username)
        password = simpledialog.askstring("Настройка администратора", "Пароль (от 12 до 1024 символов):", show="*", parent=root)
        if password is None:
            raise SystemExit("Настройка отменена. Файл .env не изменён.")
        _validate_password(password)
        confirmation = simpledialog.askstring("Настройка администратора", "Повторите пароль:", show="*", parent=root)
        if confirmation is None:
            raise SystemExit("Настройка отменена. Файл .env не изменён.")
        _validate_password(password, confirmation)
        return username, password
    finally:
        root.destroy()


def main(argv=()):
    parser = argparse.ArgumentParser(description="Первоначальная настройка администратора без сохранения открытого пароля.")
    parser.add_argument("--gui", action="store_true", help="ввод через отдельные окна с маскированными полями пароля")
    options = parser.parse_args(argv)
    from argon2 import PasswordHasher
    if options.gui:
        username, password = _gui_credentials()
    else:
        print("Пароль вводится без отображения символов. Русские и английские символы допустимы; раскладка должна совпадать при повторе.")
        try:
            username = input("Логин администратора: ").strip()
        except EOFError:
            raise SystemExit(TERMINAL_HELP) from None
        _validate_username(username)
        password = _hidden_input("Пароль (от 12 до 1024 символов): ")
        _validate_password(password)
        confirmation = _hidden_input("Повторите пароль: ")
        _validate_password(password, confirmation)
    password_hash = PasswordHasher().hash(password)
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    original = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    values = {"AUTH_ENABLED": "true", "AUTH_ADMIN_USERNAME": f"'{username}'",
              "AUTH_ADMIN_PASSWORD_HASH": f"'{password_hash}'"}
    lines = []
    for line in original.splitlines():
        key = line.strip().removeprefix("export ").split("=", 1)[0].strip()
        if key not in values:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in values.items())
    # No plaintext password or temporary plaintext credential file is written.
    with env_path.open("w", encoding="utf-8", newline="") as output:
        output.write("\n".join(lines) + "\n")
    if os.name != "nt":
        env_path.chmod(0o600)
    print("Логин и хеш пароля сохранены в .env. Авторизация включена. Перезапустите приложение.")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        raise SystemExit("Настройка прервана.") from None
