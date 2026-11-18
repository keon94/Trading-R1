// Copyright 2026 Tauric Research
// SPDX-License-Identifier: Apache-2.0

import {
  type ChatGPTAccount,
  cancelChatGPTLogin,
  chatGPTLoginStatus,
  errorText,
  getChatGPTAccounts,
  listChatGPTModels,
  selectChatGPTModel,
  signOutChatGPT,
  startChatGPTLogin,
} from "../lib/api"
import type { ListOption } from "../ui/list-dialog"
import type { Flow } from "./flow"

const TITLE = "ChatGPT account"

async function openBrowser(url: string) {
  const command = process.platform === "darwin" ? "open" : "xdg-open"
  const browser = Bun.spawn([command, url], { stdin: "ignore", stdout: "ignore", stderr: "ignore" })
  if ((await browser.exited) !== 0) throw new Error("Could not open the browser")
}

export function createChatGPT({ host, dialog, load, nav, choose, say, once }: Flow) {
  function chatgpt(back: () => void = () => dialog.close()) {
    return load(TITLE, getChatGPTAccounts(), (data) => {
      if (!data.accounts.length) {
        choose(
          TITLE,
          [{ name: "Continue with ChatGPT", description: "", value: "add" }],
          () => void signIn(null, back),
          back,
        )
        return
      }
      const options: ListOption[] = data.accounts.map((account, index) => ({
        name: `${account.email ?? "ChatGPT account"} · ${index + 1}`,
        description: [
          account.id === data.active ? "Active" : "",
          account.sharing
            ? (account.model ?? "Choose a model")
            : account.connected
              ? "Plan usage disabled"
              : "Signed out",
        ]
          .filter(Boolean)
          .join(" · "),
        value: account.id,
      }))
      options.push({ name: "Add ChatGPT account…", description: "", value: "add" })
      choose(
        TITLE,
        options,
        (id) => {
          if (id === "add") void signIn(null, back)
          else {
            const account = data.accounts.find((x) => x.id === id)
            if (account) accountActions(account, back)
          }
        },
        back,
      )
    })
  }

  function accountActions(account: ChatGPTAccount, back: () => void) {
    const options: ListOption[] = []
    if (account.sharing)
      options.push({ name: "Choose model and use account…", description: "", value: "model" })
    options.push({
      name: account.connected ? "Reconnect with ChatGPT" : "Continue with ChatGPT",
      description: "",
      value: "connect",
    })
    if (account.connected) options.push({ name: "Disconnect", description: "", value: "disconnect" })
    options.push({ name: "Manage usage in ChatGPT", description: "", value: "usage" })
    choose(
      account.email ?? TITLE,
      options,
      (action) => {
        if (action === "model") return void chooseModel(account, back)
        if (action === "connect") return void signIn(account, back)
        if (action === "usage") {
          void openBrowser("https://chatgpt.com/settings/usage").catch(() =>
            say(TITLE, ["Manage usage at https://chatgpt.com/settings/usage"], () =>
              accountActions(account, back),
            ),
          )
          return
        }
        void once(async () => {
          try {
            const { revoked } = await signOutChatGPT(account.id)
            host.onKeysChanged?.()
            say(
              TITLE,
              [
                revoked
                  ? "Disconnected from ChatGPT."
                  : "Disconnected locally. OpenAI could not confirm revocation. You can disconnect Trading-R1 in ChatGPT settings.",
              ],
              () => void chatgpt(back),
            )
          } catch (e) {
            say(TITLE, [errorText(e)], () => void chatgpt(back))
          }
        })
      },
      () => void chatgpt(back),
    )
  }

  async function signIn(account: ChatGPTAccount | null, back: () => void) {
    await once(async () => {
      const before = nav()
      let attempt: string | undefined
      let screen = before
      try {
        const login = await startChatGPTLogin(account?.id)
        attempt = login.id
        if (nav() !== before) return
        const cancel = () => void chatgpt(back)
        const wait = (manual: boolean) => {
          say(
            TITLE,
            manual
              ? ["Open this link in a browser on this machine, then return to Trading-R1.", "", login.url]
              : ["Complete sign-in in your browser and allow Trading-R1 to use your ChatGPT plan."],
            cancel,
          )
          screen = nav()
        }
        wait(false)
        void openBrowser(login.url).catch(() => {
          if (nav() === screen) wait(true)
        })
        while (nav() === screen) {
          const result = await chatGPTLoginStatus(login.id)
          if (nav() !== screen) return
          if (result.status === "error") throw new Error(result.message)
          if (result.status === "complete") {
            attempt = undefined
            host.onKeysChanged?.()
            if (result.sharing)
              return void chooseModel({ id: result.account, model: account?.model ?? null }, back)
            return say(
              TITLE,
              ["Signed in. Reconnect and approve ChatGPT plan usage to run models."],
              () => void chatgpt(back),
            )
          }
          await Bun.sleep(750)
        }
      } catch (e) {
        if (nav() === screen) say(TITLE, [errorText(e)], () => void chatgpt(back))
      } finally {
        if (attempt) void cancelChatGPTLogin(attempt).catch(() => {})
      }
    })
  }

  function chooseModel(account: Pick<ChatGPTAccount, "id" | "model">, back: () => void) {
    return load("ChatGPT models", listChatGPTModels(account.id), (models) => {
      if (!models.length) {
        return say(TITLE, ["This ChatGPT account has no available models."], () => void chatgpt(back))
      }
      choose(
        "ChatGPT models",
        models.map((model) => ({
          name: model.name,
          description: model.slug === account.model ? `${model.slug} · Current` : model.slug,
          value: model.slug,
        })),
        (model) => {
          void once(async () => {
            try {
              await selectChatGPTModel(account.id, model)
              host.onKeysChanged?.()
              host.notify(`ChatGPT model: ${model}`)
              back()
            } catch (e) {
              say(TITLE, [errorText(e)], () => void chatgpt(back))
            }
          })
        },
        () => void chatgpt(back),
      )
    })
  }

  return { chatgpt }
}
