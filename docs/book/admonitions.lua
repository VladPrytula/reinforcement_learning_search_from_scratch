--[[
Admonition Lua filter for Pandoc

Converts MkDocs-style admonitions to LaTeX tcolorbox environments.

Input (MkDocs admonition syntax):
    !!! tip "Production Checklist"
        - Item 1
        - Item 2

Output (LaTeX):
    \begin{TipBox}{Production Checklist}
    \begin{itemize}
    \item Item 1
    \item Item 2
    \end{itemize}
    \end{TipBox}

Supported admonition types:
    tip      → TipBox (green)
    note     → NoteBox (blue) - also handled by callouts.lua
    warning  → WarningBox (orange)
    danger   → DangerBox (red)
    info     → InfoBox (cyan)
    example  → ExampleBox (purple)

Note: The corresponding LaTeX environments must be defined in the template.

Usage:
    pandoc input.md --lua-filter=admonitions.lua -o output.tex
--]]

-- Map admonition types to LaTeX environment names
local ADMONITION_ENVS = {
    tip = "TipBox",
    note = "NoteBox",
    warning = "WarningBox",
    danger = "DangerBox",
    info = "InfoBox",
    example = "ExampleBox",
    important = "ImportantBox",
    caution = "CautionBox",
    abstract = "AbstractBox",
    summary = "AbstractBox",  -- alias
    question = "QuestionBox",
    quote = "QuoteBox",
}

--- Escape LaTeX special characters in a string
-- @param str string The string to escape
-- @return string The escaped string
local function escape_latex(str)
    if not str then return "" end
    -- Pandoc may stringify NonBreakingSpace as U+00A0; keep LaTeX output ASCII.
    str = str:gsub("\194\160", " ")
    local replacements = {
        ["\\"] = "\\textbackslash{}",
        ["{"] = "\\{",
        ["}"] = "\\}",
        ["%"] = "\\%",
        ["$"] = "\\$",
        ["#"] = "\\#",
        ["&"] = "\\&",
        ["_"] = "\\_",
        ["^"] = "\\^{}",
        ["~"] = "\\~{}",
    }
    return (str:gsub(".", replacements))
end

local function _is_mkdocs_marker_inline(inline)
    return inline
        and inline.t == "Str"
        and (inline.text == "!!!" or inline.text == "???")
end

local function _skip_spaces(inlines, idx)
    while idx <= #inlines and inlines[idx].t == "Space" do
        idx = idx + 1
    end
    return idx
end

local function _default_title(admon_type)
    if not admon_type or admon_type == "" then
        return ""
    end
    return admon_type:sub(1, 1):upper() .. admon_type:sub(2)
end

--- Parse an MkDocs admonition encoded as a Para.
-- Pandoc parses:
--   !!!/??? type "title"
--     body...
-- as a Para with SoftBreaks (and sometimes following CodeBlocks when blank lines exist).
--
-- @param para pandoc.Para
-- @return string|nil admonition type
-- @return string|nil title
-- @return table body inlines (after marker line)
local function parse_admonition_para(para)
    if not para or para.t ~= "Para" then
        return nil, nil, nil
    end

    local inlines = para.content
    if #inlines < 2 or not _is_mkdocs_marker_inline(inlines[1]) then
        return nil, nil, nil
    end

    local idx = 2
    idx = _skip_spaces(inlines, idx)
    if idx > #inlines or inlines[idx].t ~= "Str" then
        return nil, nil, nil
    end
    local admon_type = inlines[idx].text:lower()
    idx = idx + 1

    idx = _skip_spaces(inlines, idx)
    local title = nil
    if idx <= #inlines and inlines[idx].t == "Quoted" then
        title = pandoc.utils.stringify(inlines[idx].content)
        idx = idx + 1
    end
    if title == nil then
        title = _default_title(admon_type)
    end

    -- Body begins after the first explicit line break.
    local break_idx = nil
    for j = 1, #inlines do
        if inlines[j].t == "SoftBreak" or inlines[j].t == "LineBreak" then
            break_idx = j
            break
        end
    end

    if not break_idx then
        return admon_type, title, {}
    end

    local body_inlines = {}
    for j = break_idx + 1, #inlines do
        table.insert(body_inlines, inlines[j])
    end
    return admon_type, title, body_inlines
end

local function codeblock_has_empty_attr(block)
    if not block or block.t ~= "CodeBlock" then
        return false
    end
    local attr = block.attr
    if not attr then
        return false
    end
    local identifier = attr.identifier or ""
    local classes = attr.classes or {}
    local attributes = attr.attributes
    local has_attributes = false
    if attributes then
        for _ in pairs(attributes) do
            has_attributes = true
            break
        end
    end
    return identifier == "" and #classes == 0 and not has_attributes
end

local function inlines_to_markdown(inlines)
    if not inlines or #inlines == 0 then
        return ""
    end
    local tmp = pandoc.Pandoc({ pandoc.Para(inlines) })
    return pandoc.write(tmp, "markdown")
end

--- Process a sequence of blocks to find and convert MkDocs !!! / ??? admonitions.
-- This filter is LaTeX-only: it emits tcolorbox environments.
function Pandoc(doc)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    local new_blocks = pandoc.List()
    local i = 1

    while i <= #doc.blocks do
        local block = doc.blocks[i]

        if block.t ~= "Para" then
            new_blocks:insert(block)
            i = i + 1
            goto continue
        end

        local admon_type, title, body_inlines = parse_admonition_para(block)
        if not admon_type or not ADMONITION_ENVS[admon_type] then
            new_blocks:insert(block)
            i = i + 1
            goto continue
        end

        local env_name = ADMONITION_ENVS[admon_type]
        local escaped_title = escape_latex(title)

        -- Build markdown for the admonition body:
        -- 1) in-Para body (after marker line),
        -- 2) any immediately-following indented blocks that Pandoc parsed as CodeBlocks
        --    (these arise when the MkDocs admonition body contains blank lines).
        local body_md = inlines_to_markdown(body_inlines)
        local j = i + 1
        while j <= #doc.blocks and codeblock_has_empty_attr(doc.blocks[j]) do
            body_md = body_md .. "\n\n" .. doc.blocks[j].text .. "\n"
            j = j + 1
        end

        local body_doc = pandoc.read(body_md, "markdown")

        local begin_env = pandoc.RawBlock(
            "latex",
            "\\begin{" .. env_name .. "}{" .. escaped_title .. "}"
        )
        local end_env = pandoc.RawBlock("latex", "\\end{" .. env_name .. "}")

        new_blocks:insert(begin_env)
        for _, content_block in ipairs(body_doc.blocks) do
            new_blocks:insert(content_block)
        end
        new_blocks:insert(end_env)

        i = j
        ::continue::
    end

    return pandoc.Pandoc(new_blocks, doc.meta)
end

-- Return the filter
return {
    { Pandoc = Pandoc },
}
