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

--- Parse MkDocs admonition syntax from a Para element
-- Returns type, title, and remaining content if this is an admonition start
-- @param text string The paragraph text
-- @return string|nil admonition type
-- @return string|nil admonition title
local function parse_admonition_start(text)
    -- Pattern: !!! type "title" or !!! type 'title' or !!! type
    local admon_type, title = text:match('^!!!%s+(%w+)%s+"([^"]*)"')
    if not admon_type then
        admon_type, title = text:match("^!!!%s+(%w+)%s+'([^']*)'")
    end
    if not admon_type then
        admon_type = text:match("^!!!%s+(%w+)%s*$")
        title = admon_type and admon_type:gsub("^%l", string.upper) or nil
    end

    if admon_type then
        admon_type = admon_type:lower()
        return admon_type, title or ""
    end

    return nil, nil
end

--- Process a sequence of blocks to find and convert admonitions
-- This handles the MkDocs !!! syntax which appears as raw text
function Pandoc(doc)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    local new_blocks = pandoc.List()
    local i = 1

    while i <= #doc.blocks do
        local block = doc.blocks[i]

        -- Check if this is a Para that starts an admonition
        if block.t == "Para" then
            local text = pandoc.utils.stringify(block)
            local admon_type, title = parse_admonition_start(text)

            if admon_type and ADMONITION_ENVS[admon_type] then
                local env_name = ADMONITION_ENVS[admon_type]
                local escaped_title = escape_latex(title)

                -- Collect indented content blocks that follow
                local content_blocks = pandoc.List()
                i = i + 1

                -- Admonition content is typically in the following blocks
                -- until we hit a non-indented block or another admonition
                while i <= #doc.blocks do
                    local next_block = doc.blocks[i]

                    -- Check if next block is another admonition or unindented content
                    if next_block.t == "Para" then
                        local next_text = pandoc.utils.stringify(next_block)
                        if next_text:match("^!!!") then
                            -- Another admonition, stop here
                            break
                        end
                    end

                    -- For simplicity, include the next block as content
                    -- In practice, MkDocs admonitions use indentation which
                    -- Pandoc may parse differently
                    content_blocks:insert(next_block)
                    i = i + 1

                    -- Only take one block of content for now
                    -- (MkDocs admonitions are typically single-block)
                    break
                end

                -- Create the LaTeX wrapper
                local begin_env = pandoc.RawBlock("latex",
                    "\\begin{" .. env_name .. "}{" .. escaped_title .. "}")
                local end_env = pandoc.RawBlock("latex",
                    "\\end{" .. env_name .. "}")

                new_blocks:insert(begin_env)
                for _, content_block in ipairs(content_blocks) do
                    new_blocks:insert(content_block)
                end
                new_blocks:insert(end_env)
            else
                -- Not an admonition, keep as is
                new_blocks:insert(block)
                i = i + 1
            end
        else
            -- Not a Para, keep as is
            new_blocks:insert(block)
            i = i + 1
        end
    end

    return pandoc.Pandoc(new_blocks, doc.meta)
end

-- Return the filter
return {
    { Pandoc = Pandoc },
}
