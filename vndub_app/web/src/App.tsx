import { Suspense, lazy, useEffect, type ReactNode } from "react";
import { HashRouter, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { ErrorPanel } from "./components/ErrorPanel";
import { ActiveJobBar, ResumePrompt, Stepper, TopBar } from "./components/Shell";
import { ToastProvider } from "./components/ui";
import { STEP_IDS, type StepId } from "./api/types";
import { useI18n } from "./i18n";
import { ProjectProvider, useProject } from "./state/project";
import { startProgress } from "./state/progress";
import { SettingsProvider } from "./state/settings";
import { stepPath } from "./state/wizard";
import { firstRunDone } from "./util/firstRun";

const DoctorScreen = lazy(() => import("./screens/DoctorScreen"));
const GameScreen = lazy(() => import("./screens/GameScreen"));
const TranslateScreen = lazy(() => import("./screens/TranslateScreen"));
const CastingScreen = lazy(() => import("./screens/CastingScreen"));
const PreviewScreen = lazy(() => import("./screens/PreviewScreen"));
const RunScreen = lazy(() => import("./screens/RunScreen"));
const ReviewScreen = lazy(() => import("./screens/ReviewScreen"));
const InstallScreen = lazy(() => import("./screens/InstallScreen"));
const ModelsScreen = lazy(() => import("./screens/ModelsScreen"));
const SettingsScreen = lazy(() => import("./screens/SettingsScreen"));
const HelpScreen = lazy(() => import("./screens/HelpScreen"));

function Loading() {
  const t = useI18n().t;
  return (
    <p className="loading" role="status">
      {t("app.loading")}
    </p>
  );
}

const SCREENS: Record<StepId, () => ReactNode> = {
  doctor: () => <DoctorScreen />,
  game: () => <GameScreen />,
  translate: () => <TranslateScreen />,
  casting: () => <CastingScreen />,
  preview: () => <PreviewScreen />,
  run: () => <RunScreen />,
  review: () => <ReviewScreen />,
  install: () => <InstallScreen />,
};

/** Move keyboard focus to the page heading after navigation (screen readers announce the new page). */
function useFocusHeading() {
  const loc = useLocation();
  useEffect(() => {
    const id = window.setTimeout(() => {
      const h = document.querySelector<HTMLElement>("main h1");
      h?.focus({ preventScroll: true });
    }, 60);
    return () => window.clearTimeout(id);
  }, [loc.pathname]);
}

function Page({ children, currentStep }: { children: ReactNode; currentStep: StepId | null }) {
  const t = useI18n().t;
  useFocusHeading();
  return (
    <div className="app">
      <a href="#main" className="skip-link" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>
        {t("app.skip")}
      </a>
      <TopBar />
      <ActiveJobBar currentStep={currentStep} />
      {currentStep ? <Stepper current={currentStep} /> : null}
      <main id="main" tabIndex={-1} className="main">
        <Suspense fallback={<Loading />}>{children}</Suspense>
      </main>
      <ResumePrompt />
    </div>
  );
}

function StepRoute() {
  const { step: param } = useParams();
  const { project, saveState, loading } = useProject();
  const step = (STEP_IDS as string[]).includes(param ?? "") ? (param as StepId) : null;

  useEffect(() => {
    if (step && project && project.state.step !== step) void saveState({ step });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, project?.slug]);

  if (loading) return <Loading />;
  if (!step) return <Navigate to="/" replace />;
  // Without a game open only Diagnostics and the game picker make sense.
  if (!project && step !== "doctor" && step !== "game") return <Navigate to={stepPath("game")} replace />;
  return (
    <Page currentStep={step}>
      <div key={step} className="screen" data-testid={`screen-${step}`}>
        {SCREENS[step]()}
      </div>
    </Page>
  );
}

function Landing() {
  const { project, loading, error, reload } = useProject();
  if (loading) return <Loading />;
  if (error) {
    return (
      <Page currentStep={null}>
        <ErrorPanel error={error} onRetry={() => void reload()} />
      </Page>
    );
  }
  if (!project) return <Navigate to={stepPath(firstRunDone() ? "game" : "doctor")} replace />;
  return <Navigate to={stepPath(project.state.step)} replace />;
}

function Plain({ children }: { children: ReactNode }) {
  return <Page currentStep={null}>{children}</Page>;
}

export default function App() {
  useEffect(() => {
    startProgress();
  }, []);
  return (
    <HashRouter>
      <SettingsProvider>
        <ToastProvider>
          <ProjectProvider>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/step/:step" element={<StepRoute />} />
              <Route path="/models" element={<Plain><ModelsScreen tab="models" /></Plain>} />
              <Route path="/tools" element={<Plain><ModelsScreen tab="tools" /></Plain>} />
              <Route path="/settings" element={<Plain><SettingsScreen /></Plain>} />
              <Route path="/help" element={<Plain><HelpScreen /></Plain>} />
              <Route path="/help/:id" element={<Plain><HelpScreen /></Plain>} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ProjectProvider>
        </ToastProvider>
      </SettingsProvider>
    </HashRouter>
  );
}
